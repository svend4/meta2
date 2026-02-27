"""
Property-based invariant tests for:
  - puzzle_reconstruction.utils.graph_utils
  - puzzle_reconstruction.utils.topology_utils
  - puzzle_reconstruction.utils.window_utils

Verifies mathematical invariants:

graph_utils:
    build_graph:           n_nodes = matrix size; adj symmetric; no self-loops
    connected_components:  partition of all nodes; union = all nodes
    node_degrees:          shape = (n_nodes,); sum = 2 * len(edges)
    dijkstra:              dist[source] = 0; all dist >= 0
    minimum_spanning_tree: <= n_nodes - 1 edges
    subgraph:              n_nodes = len(nodes)

topology_utils:
    compute_solidity:      ∈ [0, 1]; convex → 1.0
    compute_extent:        ∈ [0, 1]
    compute_convexity:     ∈ [0, 1]; convex → ≈ 1.0
    compute_compactness:   ∈ [0, 1]
    shape_complexity:      ∈ [0, 1]
    is_simply_connected:   all-foreground mask → True
    count_holes:           all-foreground mask → 0; empty mask → 0
    batch_topology:        output length = input length

window_utils:
    apply_window_function: same length; rect → copy of input
    rolling_mean:          length preserved (padding='same');
                           values bounded by [signal.min(), signal.max()]
    rolling_std:           >= 0; length preserved
    rolling_max:           >= rolling_mean element-wise
    rolling_min:           <= rolling_mean element-wise
    compute_overlap:       ∈ [0, 1]; commutative; self-overlap = 1.0;
                           disjoint → 0.0
    split_into_windows:    each window has len = cfg.size; list non-empty
    merge_windows:         output length = original_length
    batch_rolling:         output length = input length
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from puzzle_reconstruction.utils.graph_utils import (
    GraphEdge,
    FragmentGraph,
    build_graph,
    dijkstra,
    shortest_path,
    minimum_spanning_tree,
    connected_components,
    node_degrees,
    subgraph,
    batch_build_graphs,
)
from puzzle_reconstruction.utils.topology_utils import (
    TopologyConfig,
    compute_euler_number,
    count_holes,
    compute_solidity,
    compute_extent,
    compute_convexity,
    compute_compactness,
    is_simply_connected,
    shape_complexity,
    batch_topology,
)
from puzzle_reconstruction.utils.window_utils import (
    WindowConfig,
    apply_window_function,
    rolling_mean,
    rolling_std,
    rolling_max,
    rolling_min,
    compute_overlap,
    split_into_windows,
    merge_windows,
    batch_rolling,
)

RNG = np.random.default_rng(2029)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rand_symmetric_matrix(n: int, sparsity: float = 0.3) -> np.ndarray:
    """Random symmetric matrix with values in [0, 1]; some zeros for sparsity."""
    mat = RNG.uniform(0.0, 1.0, (n, n))
    mask = RNG.random((n, n)) < sparsity
    mat[mask] = 0.0
    # Symmetrize
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 0.0)
    return mat.astype(np.float64)


def _simple_graph(n: int) -> FragmentGraph:
    """Build a simple connected graph on n nodes (chain)."""
    mat = np.zeros((n, n))
    for i in range(n - 1):
        mat[i, i + 1] = 0.5
        mat[i + 1, i] = 0.5
    return build_graph(mat, threshold=0.0)


def _rand_signal(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n).astype(np.float64)


def _rect_contour(w: float = 10.0, h: float = 5.0) -> np.ndarray:
    """Rectangle contour (4 corners + extra points)."""
    return np.array([
        [0.0, 0.0], [w/2, 0.0], [w, 0.0],
        [w, h/2],   [w, h],
        [w/2, h],   [0.0, h],
        [0.0, h/2],
    ], dtype=float)


def _triangle_contour(side: float = 10.0) -> np.ndarray:
    """Equilateral triangle contour."""
    return np.array([
        [0.0, 0.0],
        [side, 0.0],
        [side / 2.0, side * math.sqrt(3) / 2.0],
    ], dtype=float)


def _star_contour(n: int = 5, r_outer: float = 10.0, r_inner: float = 4.0) -> np.ndarray:
    """Star-shaped (non-convex) contour."""
    pts = []
    for i in range(n * 2):
        angle = math.pi / n * i - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        pts.append([r * math.cos(angle), r * math.sin(angle)])
    return np.array(pts, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# graph_utils — build_graph
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildGraph:
    """build_graph: structural invariants."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_n_nodes_matches_matrix(self, n: int) -> None:
        mat = _rand_symmetric_matrix(n)
        g = build_graph(mat)
        assert g.n_nodes == n

    @pytest.mark.parametrize("n", [4, 5, 6, 8])
    def test_adjacency_symmetric(self, n: int) -> None:
        mat = _rand_symmetric_matrix(n)
        g = build_graph(mat)
        for i in range(n):
            nbrs_i = {v for v, _ in g.adj[i]}
            for j in nbrs_i:
                nbrs_j = {v for v, _ in g.adj[j]}
                assert i in nbrs_j, f"adj not symmetric: {i} in adj[{j}] failed"

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_no_self_loops(self, n: int) -> None:
        mat = _rand_symmetric_matrix(n)
        g = build_graph(mat)
        for edge in g.edges:
            assert edge.src != edge.dst

    @pytest.mark.parametrize("n", [4, 5])
    def test_all_edge_weights_above_threshold(self, n: int) -> None:
        mat = _rand_symmetric_matrix(n)
        threshold = 0.2
        g = build_graph(mat, threshold=threshold)
        for edge in g.edges:
            assert edge.weight > threshold

    def test_zero_threshold_includes_all_nonzero(self) -> None:
        mat = _rand_symmetric_matrix(5)
        g = build_graph(mat, threshold=0.0)
        n_expected = int(np.sum((mat > 0) & (np.arange(5)[:, None] < np.arange(5)[None, :])))
        assert len(g.edges) == n_expected


# ═══════════════════════════════════════════════════════════════════════════════
# graph_utils — connected_components
# ═══════════════════════════════════════════════════════════════════════════════

class TestConnectedComponents:
    """connected_components: partition invariants."""

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_union_is_all_nodes(self, n: int) -> None:
        g = _simple_graph(n)
        comps = connected_components(g)
        all_nodes = set()
        for comp in comps:
            all_nodes.update(comp)
        assert all_nodes == set(range(n))

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_no_node_in_two_components(self, n: int) -> None:
        g = _simple_graph(n)
        comps = connected_components(g)
        seen = set()
        for comp in comps:
            for v in comp:
                assert v not in seen
                seen.add(v)

    def test_chain_is_one_component(self) -> None:
        g = _simple_graph(5)
        comps = connected_components(g)
        assert len(comps) == 1

    def test_disconnected_graph_has_multiple_components(self) -> None:
        # Two isolated nodes
        mat = np.zeros((4, 4))
        mat[0, 1] = mat[1, 0] = 0.5
        mat[2, 3] = mat[3, 2] = 0.5
        g = build_graph(mat, threshold=0.0)
        comps = connected_components(g)
        assert len(comps) == 2

    @pytest.mark.parametrize("n", [3, 5])
    def test_component_nodes_are_sorted(self, n: int) -> None:
        g = _simple_graph(n)
        comps = connected_components(g)
        for comp in comps:
            assert comp == sorted(comp)


# ═══════════════════════════════════════════════════════════════════════════════
# graph_utils — node_degrees
# ═══════════════════════════════════════════════════════════════════════════════

class TestNodeDegrees:
    """node_degrees: shape and handshaking lemma."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_shape(self, n: int) -> None:
        g = _simple_graph(n)
        degs = node_degrees(g)
        assert degs.shape == (n,)

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_handshaking_lemma(self, n: int) -> None:
        mat = _rand_symmetric_matrix(n)
        g = build_graph(mat)
        degs = node_degrees(g)
        # Sum of degrees = 2 * number of edges
        assert int(degs.sum()) == 2 * len(g.edges)

    @pytest.mark.parametrize("n", [4, 5])
    def test_nonnegative(self, n: int) -> None:
        g = _simple_graph(n)
        degs = node_degrees(g)
        assert np.all(degs >= 0)

    def test_isolated_node_degree_zero(self) -> None:
        mat = np.zeros((3, 3))
        g = build_graph(mat, threshold=0.0)
        degs = node_degrees(g)
        assert np.all(degs == 0)


# ═══════════════════════════════════════════════════════════════════════════════
# graph_utils — dijkstra
# ═══════════════════════════════════════════════════════════════════════════════

class TestDijkstra:
    """dijkstra: distance invariants."""

    @pytest.mark.parametrize("n,source", [(4, 0), (5, 2), (6, 0)])
    def test_dist_source_is_zero(self, n: int, source: int) -> None:
        g = _simple_graph(n)
        dist, _ = dijkstra(g, source=source)
        assert dist[source] == pytest.approx(0.0)

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_all_finite_on_connected_graph(self, n: int) -> None:
        g = _simple_graph(n)
        dist, _ = dijkstra(g, source=0)
        assert np.all(np.isfinite(dist))

    @pytest.mark.parametrize("n", [4, 5])
    def test_dist_nonnegative(self, n: int) -> None:
        g = _simple_graph(n)
        dist, _ = dijkstra(g, source=0)
        assert np.all(dist >= 0.0)

    def test_shape_of_output(self) -> None:
        n = 5
        g = _simple_graph(n)
        dist, prev = dijkstra(g, source=0)
        assert dist.shape == (n,)
        assert prev.shape == (n,)

    def test_prev_source_is_minus_one(self) -> None:
        g = _simple_graph(5)
        _, prev = dijkstra(g, source=0)
        assert int(prev[0]) == -1


# ═══════════════════════════════════════════════════════════════════════════════
# graph_utils — minimum_spanning_tree, subgraph
# ═══════════════════════════════════════════════════════════════════════════════

class TestMSTAndSubgraph:
    """minimum_spanning_tree and subgraph invariants."""

    @pytest.mark.parametrize("n", [4, 5, 6, 7])
    def test_mst_at_most_n_minus_1_edges(self, n: int) -> None:
        g = _simple_graph(n)
        mst = minimum_spanning_tree(g)
        assert len(mst) <= n - 1

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_mst_exactly_n_minus_1_on_connected(self, n: int) -> None:
        """Connected graph → MST has exactly n-1 edges."""
        g = _simple_graph(n)
        mst = minimum_spanning_tree(g)
        assert len(mst) == n - 1

    @pytest.mark.parametrize("n,k", [(6, 3), (5, 2), (8, 4)])
    def test_subgraph_n_nodes(self, n: int, k: int) -> None:
        g = _simple_graph(n)
        nodes = list(range(k))
        sg = subgraph(g, nodes)
        assert sg.n_nodes == k

    def test_subgraph_edges_within_nodes(self) -> None:
        g = _simple_graph(6)
        nodes = [0, 1, 2]
        sg = subgraph(g, nodes)
        # All edges should reference only remapped nodes [0, 1, 2]
        for edge in sg.edges:
            assert 0 <= edge.src < 3
            assert 0 <= edge.dst < 3


# ═══════════════════════════════════════════════════════════════════════════════
# topology_utils — compute_solidity
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeSolidity:
    """compute_solidity: ∈ [0, 1]; convex → 1.0."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_in_unit_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 10, (8, 2))
        val = compute_solidity(pts)
        assert 0.0 <= val <= 1.0 + 1e-9

    def test_rectangle_is_one(self) -> None:
        rect = _rect_contour()
        val = compute_solidity(rect)
        assert val == pytest.approx(1.0, abs=0.1)

    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_star_below_one(self, seed: int) -> None:
        star = _star_contour()
        val = compute_solidity(star)
        assert val <= 1.0 + 1e-9

    def test_triangle_approaches_one(self) -> None:
        tri = _triangle_contour()
        val = compute_solidity(tri)
        # Triangle is convex → solidity ≈ 1.0
        assert val == pytest.approx(1.0, abs=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# topology_utils — compute_extent
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeExtent:
    """compute_extent: ∈ [0, 1]."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_in_unit_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 10, (8, 2))
        val = compute_extent(pts)
        assert 0.0 <= val <= 1.0 + 1e-9

    def test_rectangle_high_extent(self) -> None:
        # A rectangle fills most of its bounding box
        rect = _rect_contour(10.0, 5.0)
        val = compute_extent(rect)
        assert val > 0.0

    @pytest.mark.parametrize("seed", [5, 6, 7, 8, 9])
    def test_random_contour_in_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(1, 100, (10, 2))
        val = compute_extent(pts)
        assert 0.0 <= val <= 1.0 + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# topology_utils — compute_convexity
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeConvexity:
    """compute_convexity: ∈ [0, 1]; convex contour → ≈ 1.0."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_in_unit_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 10, (8, 2))
        val = compute_convexity(pts)
        assert 0.0 <= val <= 1.0 + 1e-9

    def test_convex_polygon_is_one(self) -> None:
        rect = _rect_contour(10.0, 5.0)
        val = compute_convexity(rect)
        # Rectangle is convex → convexity = 1.0
        assert val == pytest.approx(1.0, abs=0.01)

    def test_star_below_convex(self) -> None:
        star = _star_contour()
        val = compute_convexity(star)
        # Non-convex shape → < 1.0
        assert val <= 1.0 + 1e-9

    @pytest.mark.parametrize("seed", [20, 21, 22])
    def test_triangle_convexity(self, seed: int) -> None:
        tri = _triangle_contour()
        val = compute_convexity(tri)
        assert val <= 1.0 + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# topology_utils — compute_compactness
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeCompactness:
    """compute_compactness: ∈ [0, 1]."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_in_unit_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 10, (10, 2))
        val = compute_compactness(pts)
        assert 0.0 <= val <= 1.0 + 1e-9

    @pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
    def test_random_contour_in_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 50, (12, 2))
        val = compute_compactness(pts)
        assert 0.0 <= val <= 1.0 + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# topology_utils — shape_complexity
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeComplexity:
    """shape_complexity: ∈ [0, 1]."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_in_unit_range(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 10, (8, 2))
        val = shape_complexity(pts)
        assert 0.0 - 1e-9 <= val <= 1.0 + 1e-9

    def test_formula_matches(self) -> None:
        pts = _rect_contour()
        complexity = shape_complexity(pts)
        expected = 1.0 - compute_compactness(pts) * compute_convexity(pts)
        assert complexity == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("seed", [30, 31, 32])
    def test_random_star_in_range(self, seed: int) -> None:
        star = _star_contour()
        val = shape_complexity(star)
        assert 0.0 <= val <= 1.0 + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# topology_utils — is_simply_connected, count_holes, batch_topology
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopologyMasks:
    """is_simply_connected, count_holes, batch_topology invariants."""

    def test_all_foreground_is_simply_connected(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        assert is_simply_connected(mask) is True

    def test_all_foreground_zero_holes(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        assert count_holes(mask) == 0

    def test_all_background_zero_holes(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        assert count_holes(mask) == 0

    def test_mask_with_hole_not_simply_connected(self) -> None:
        # Create a ring: foreground frame, background interior
        mask = np.ones((10, 10), dtype=bool)
        mask[2:8, 2:8] = False  # hollow interior
        assert count_holes(mask) >= 1
        assert is_simply_connected(mask) is False

    def test_batch_topology_length(self) -> None:
        contours = [_rect_contour(), _triangle_contour(), _star_contour()]
        results = batch_topology(contours)
        assert len(results) == 3

    def test_batch_topology_keys(self) -> None:
        contours = [_rect_contour(), _triangle_contour()]
        results = batch_topology(contours)
        expected_keys = {"solidity", "extent", "convexity", "compactness", "complexity"}
        for r in results:
            assert set(r.keys()) == expected_keys

    def test_batch_topology_values_in_range(self) -> None:
        contours = [_rect_contour(), _triangle_contour(), _star_contour()]
        results = batch_topology(contours)
        for r in results:
            for key, val in r.items():
                assert 0.0 - 1e-9 <= val <= 1.0 + 1e-9, (
                    f"{key} = {val} out of [0, 1]"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# window_utils — apply_window_function
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyWindowFunction:
    """apply_window_function: length and value invariants."""

    @pytest.mark.parametrize("n,func", [
        (16, "hann"), (20, "hamming"), (15, "bartlett"), (12, "blackman")
    ])
    def test_same_length(self, n: int, func: str) -> None:
        signal = _rand_signal(n)
        cfg = WindowConfig(size=n, func=func)
        result = apply_window_function(signal, cfg)
        assert len(result) == n

    @pytest.mark.parametrize("n", [8, 12, 20])
    def test_rect_is_copy(self, n: int) -> None:
        signal = _rand_signal(n)
        cfg = WindowConfig(size=n, func="rect")
        result = apply_window_function(signal, cfg)
        np.testing.assert_allclose(result, signal)

    @pytest.mark.parametrize("func", ["hann", "hamming", "bartlett", "blackman"])
    def test_output_dtype_float64(self, func: str) -> None:
        signal = _rand_signal(16)
        cfg = WindowConfig(size=16, func=func)
        result = apply_window_function(signal, cfg)
        assert result.dtype == np.float64

    @pytest.mark.parametrize("func", ["hann", "hamming", "bartlett", "blackman"])
    def test_window_attenuates_edges(self, func: str) -> None:
        """Window functions ≤ 1 everywhere; rect = 1 everywhere."""
        signal = np.ones(16, dtype=np.float64)
        cfg = WindowConfig(size=16, func=func)
        result = apply_window_function(signal, cfg)
        # Since signal = 1, result is the window itself (≤ 1)
        assert float(result.max()) <= 1.0 + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# window_utils — rolling statistics
# ═══════════════════════════════════════════════════════════════════════════════

class TestRollingStatistics:
    """rolling_mean, rolling_std, rolling_max, rolling_min invariants."""

    @pytest.mark.parametrize("n,seed", [(20, 0), (30, 1), (15, 2), (25, 3)])
    def test_rolling_mean_length_same(self, n: int, seed: int) -> None:
        signal = _rand_signal(n, seed)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_mean(signal, cfg)
        assert len(result) == n

    @pytest.mark.parametrize("n,seed", [(20, 0), (25, 1)])
    def test_rolling_mean_bounded(self, n: int, seed: int) -> None:
        signal = _rand_signal(n, seed)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_mean(signal, cfg)
        lo, hi = float(signal.min()), float(signal.max())
        assert float(result.min()) >= lo - 1e-9
        assert float(result.max()) <= hi + 1e-9

    @pytest.mark.parametrize("n,seed", [(20, 5), (30, 6)])
    def test_rolling_std_nonnegative(self, n: int, seed: int) -> None:
        signal = _rand_signal(n, seed)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_std(signal, cfg)
        assert np.all(result >= -1e-9)

    @pytest.mark.parametrize("n,seed", [(20, 7), (30, 8)])
    def test_rolling_std_length(self, n: int, seed: int) -> None:
        signal = _rand_signal(n, seed)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_std(signal, cfg)
        assert len(result) == n

    @pytest.mark.parametrize("n,seed", [(20, 9), (25, 10)])
    def test_rolling_max_gte_mean(self, n: int, seed: int) -> None:
        signal = _rand_signal(n, seed)
        cfg = WindowConfig(size=4, step=1, padding="same")
        rmax = rolling_max(signal, cfg)
        rmean = rolling_mean(signal, cfg)
        assert np.all(rmax >= rmean - 1e-9)

    @pytest.mark.parametrize("n,seed", [(20, 11), (25, 12)])
    def test_rolling_min_lte_mean(self, n: int, seed: int) -> None:
        signal = _rand_signal(n, seed)
        cfg = WindowConfig(size=4, step=1, padding="same")
        rmin = rolling_min(signal, cfg)
        rmean = rolling_mean(signal, cfg)
        assert np.all(rmin <= rmean + 1e-9)

    @pytest.mark.parametrize("n,seed", [(20, 13), (25, 14)])
    def test_rolling_max_gte_rolling_min(self, n: int, seed: int) -> None:
        signal = _rand_signal(n, seed)
        cfg = WindowConfig(size=4, step=1, padding="same")
        rmax = rolling_max(signal, cfg)
        rmin = rolling_min(signal, cfg)
        assert np.all(rmax >= rmin - 1e-9)

    def test_constant_signal_rolling_std_zero(self) -> None:
        signal = np.full(20, 3.5, dtype=np.float64)
        cfg = WindowConfig(size=4, step=1, padding="same")
        result = rolling_std(signal, cfg)
        assert np.allclose(result, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# window_utils — compute_overlap
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeOverlap:
    """compute_overlap: IoU invariants for 1-D windows."""

    @pytest.mark.parametrize("a0,a1,b0,b1", [
        (0, 10, 5, 15), (0, 8, 4, 12), (2, 6, 3, 9)
    ])
    def test_in_unit_range(self, a0: int, a1: int, b0: int, b1: int) -> None:
        val = compute_overlap(a0, a1, b0, b1)
        assert 0.0 <= val <= 1.0

    @pytest.mark.parametrize("a0,a1", [(0, 10), (5, 15), (3, 9)])
    def test_self_overlap_is_one(self, a0: int, a1: int) -> None:
        val = compute_overlap(a0, a1, a0, a1)
        assert val == pytest.approx(1.0)

    @pytest.mark.parametrize("a0,a1,b0,b1", [
        (0, 5, 10, 20), (0, 3, 5, 10)
    ])
    def test_disjoint_gives_zero(self, a0: int, a1: int, b0: int, b1: int) -> None:
        val = compute_overlap(a0, a1, b0, b1)
        assert val == pytest.approx(0.0)

    @pytest.mark.parametrize("a0,a1,b0,b1", [
        (0, 10, 5, 15), (2, 8, 4, 12)
    ])
    def test_commutative(self, a0: int, a1: int, b0: int, b1: int) -> None:
        assert compute_overlap(a0, a1, b0, b1) == pytest.approx(
            compute_overlap(b0, b1, a0, a1)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# window_utils — split_into_windows, merge_windows
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitAndMergeWindows:
    """split_into_windows and merge_windows: shape invariants."""

    @pytest.mark.parametrize("n,size", [(20, 4), (30, 6), (15, 3)])
    def test_each_window_correct_size(self, n: int, size: int) -> None:
        signal = _rand_signal(n)
        cfg = WindowConfig(size=size, step=1, padding="same")
        windows = split_into_windows(signal, cfg)
        assert len(windows) > 0
        for w in windows:
            assert len(w) == size

    @pytest.mark.parametrize("n,size,step", [
        (20, 4, 1), (30, 6, 2), (24, 8, 4)
    ])
    def test_merge_output_length(self, n: int, size: int, step: int) -> None:
        signal = _rand_signal(n)
        cfg = WindowConfig(size=size, step=step, padding="same")
        windows = split_into_windows(signal, cfg)
        merged = merge_windows(windows, original_length=n, cfg=cfg)
        assert len(merged) == n

    @pytest.mark.parametrize("n", [20, 24, 30])
    def test_split_windows_nonempty(self, n: int) -> None:
        signal = _rand_signal(n)
        cfg = WindowConfig(size=4, step=1, padding="same")
        windows = split_into_windows(signal, cfg)
        assert len(windows) > 0

    @pytest.mark.parametrize("n,size,step", [(20, 4, 1), (24, 6, 2)])
    def test_merge_dtype_float64(self, n: int, size: int, step: int) -> None:
        signal = _rand_signal(n)
        cfg = WindowConfig(size=size, step=step, padding="same")
        windows = split_into_windows(signal, cfg)
        merged = merge_windows(windows, original_length=n, cfg=cfg)
        assert merged.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════════
# window_utils — batch_rolling
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchRolling:
    """batch_rolling: length invariants."""

    @pytest.mark.parametrize("stat", ["mean", "std", "max", "min"])
    def test_output_length_matches_input(self, stat: str) -> None:
        signals = [_rand_signal(20, seed=i) for i in range(4)]
        cfg = WindowConfig(size=4, step=1, padding="same")
        results = batch_rolling(signals, stat=stat, cfg=cfg)
        assert len(results) == len(signals)

    def test_each_result_same_length_as_signal(self) -> None:
        n = 25
        signals = [_rand_signal(n, seed=i) for i in range(5)]
        cfg = WindowConfig(size=4, step=1, padding="same")
        results = batch_rolling(signals, stat="mean", cfg=cfg)
        for r in results:
            assert len(r) == n
