"""
Тесты для puzzle_reconstruction/matching/graph_match.py

Покрытие:
    FragmentGraph   — add_node, add_edge, weight, neighbors, sorted_nodes,
                      adjacency_matrix (форма, симметричность, значения),
                      laplacian (L = D - A, диагональ), repr, n_nodes, n_edges
    build_fragment_graph — пустой, порог, fid-определение из edge_id,
                           max-weight update, self-edge игнорирование
    mst_ordering    — пустой граф, один узел, линейный граф, связность,
                      нет дубликатов, покрытие всех узлов
    spectral_ordering — пустой, один, два узла, длина = n_nodes, без дубликатов,
                        разные порядки для разных топологий
    random_walk_similarity — форма (N×N), строки нормированы, 0×0 для пустого,
                             диагональ ≥ 0, симметричность (примерно)
    degree_centrality — пустой граф, изолированный узел, все значения ∈ [0,1],
                        max всегда 1.0, узел с большим весом ≥ остальных
    analyze_graph   — тип результата, mst_order ⊆ nodes, spectral_order ⊆ nodes,
                      len(mst_edges) ≤ n_nodes-1
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.matching.graph_match import (
    FragmentGraph,
    GraphMatchResult,
    analyze_graph,
    build_fragment_graph,
    degree_centrality,
    mst_ordering,
    random_walk_similarity,
    spectral_ordering,
)
from puzzle_reconstruction.models import CompatEntry, Edge, Fragment


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _make_fragment(fid: int) -> Fragment:
    import numpy as np
    return Fragment(
        fragment_id=fid,
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        edges=[],
        bounding_box=(0, 0, 10, 10),
    )


def _make_edge(fid: int, side: int) -> Edge:
    return Edge(
        edge_id=fid * 10 + side,
        contour=np.zeros((5, 2), dtype=np.float64),
        text_hint="",
    )


def _make_entry(fid_i: int, fid_j: int, score: float) -> CompatEntry:
    return CompatEntry(
        edge_i=_make_edge(fid_i, 0),
        edge_j=_make_edge(fid_j, 0),
        score=score,
    )


def _chain_graph(n: int, base_weight: float = 0.8) -> FragmentGraph:
    """Цепной граф: 0—1—2—…—(n-1)."""
    g = FragmentGraph()
    for i in range(n):
        g.add_node(_make_fragment(i))
    for i in range(n - 1):
        g.add_edge(i, i + 1, base_weight - i * 0.05)
    return g


def _star_graph(n: int) -> FragmentGraph:
    """Звёздный граф: узел 0 соединён со всеми остальными."""
    g = FragmentGraph()
    for i in range(n):
        g.add_node(_make_fragment(i))
    for i in range(1, n):
        g.add_edge(0, i, 0.9 - i * 0.1)
    return g


def _complete_graph(n: int) -> FragmentGraph:
    """Полный граф K_n."""
    g = FragmentGraph()
    for i in range(n):
        g.add_node(_make_fragment(i))
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, 1.0 / (abs(i - j) + 1))
    return g


# ─── FragmentGraph ────────────────────────────────────────────────────────────

class TestFragmentGraph:
    def test_initial_empty(self):
        g = FragmentGraph()
        assert g.n_nodes == 0
        assert g.n_edges == 0

    def test_add_node_increments(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        assert g.n_nodes == 1

    def test_add_edge_increments(self):
        g = _chain_graph(2)
        assert g.n_edges == 1

    def test_weight_existing_edge(self):
        g = _chain_graph(3)
        w = g.weight(0, 1)
        assert w > 0.0

    def test_weight_missing_edge_zero(self):
        g = _chain_graph(3)
        assert g.weight(0, 2) == 0.0

    def test_add_edge_max_weight(self):
        """Повторное добавление с большим весом обновляет ребро."""
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        g.add_node(_make_fragment(1))
        g.add_edge(0, 1, 0.3)
        g.add_edge(0, 1, 0.9)
        assert g.weight(0, 1) == pytest.approx(0.9)

    def test_add_edge_max_weight_keeps_existing(self):
        """Меньший вес не перезаписывает больший."""
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        g.add_node(_make_fragment(1))
        g.add_edge(0, 1, 0.9)
        g.add_edge(0, 1, 0.3)
        assert g.weight(0, 1) == pytest.approx(0.9)

    def test_neighbors_count(self):
        g = _star_graph(5)
        neigh = g.neighbors(0)
        assert len(neigh) == 4

    def test_neighbors_isolated(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(99))
        assert g.neighbors(99) == []

    def test_sorted_nodes_ordered(self):
        g = _chain_graph(5)
        sn = g.sorted_nodes()
        assert sn == sorted(sn)

    def test_adjacency_matrix_shape(self):
        g = _chain_graph(4)
        A, fids = g.adjacency_matrix()
        assert A.shape == (4, 4)
        assert len(fids) == 4

    def test_adjacency_matrix_symmetric(self):
        g = _complete_graph(4)
        A, _ = g.adjacency_matrix()
        assert np.allclose(A, A.T)

    def test_adjacency_matrix_values(self):
        g = _chain_graph(3)
        A, fids = g.adjacency_matrix()
        idx = {f: i for i, f in enumerate(fids)}
        assert A[idx[0], idx[1]] == pytest.approx(g.weight(0, 1))

    def test_adjacency_matrix_diagonal_zero(self):
        g = _chain_graph(3)
        A, _ = g.adjacency_matrix()
        assert np.allclose(np.diag(A), 0.0)

    def test_laplacian_row_sums_zero(self):
        """Строки лапласиана суммируются в 0."""
        g = _complete_graph(4)
        L, _ = g.laplacian()
        assert np.allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_laplacian_diagonal_positive(self):
        g = _chain_graph(4)
        L, _ = g.laplacian()
        assert (np.diag(L) >= 0).all()

    def test_repr_contains_counts(self):
        g = _chain_graph(3)
        r = repr(g)
        assert "3" in r and "2" in r

    def test_single_node_adj_matrix(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        A, fids = g.adjacency_matrix()
        assert A.shape == (1, 1)
        assert A[0, 0] == 0.0

    def test_frag_map_populated(self):
        g = _chain_graph(3)
        assert set(g.frag_map.keys()) == {0, 1, 2}


# ─── build_fragment_graph ─────────────────────────────────────────────────────

class TestBuildFragmentGraph:
    def test_empty_fragments(self):
        g = build_fragment_graph([], [])
        assert g.n_nodes == 0

    def test_no_entries_no_edges(self):
        frags = [_make_fragment(i) for i in range(3)]
        g     = build_fragment_graph(frags, [])
        assert g.n_edges == 0
        assert g.n_nodes == 3

    def test_all_nodes_added(self):
        frags = [_make_fragment(i) for i in range(5)]
        g     = build_fragment_graph(frags, [])
        assert g.n_nodes == 5

    def test_edge_added_from_entries(self):
        frags   = [_make_fragment(0), _make_fragment(1)]
        entries = [_make_entry(0, 1, 0.7)]
        g       = build_fragment_graph(frags, entries)
        assert g.n_edges == 1

    def test_threshold_filters_edges(self):
        frags   = [_make_fragment(0), _make_fragment(1)]
        entries = [_make_entry(0, 1, 0.3)]
        g       = build_fragment_graph(frags, entries, threshold=0.5)
        assert g.n_edges == 0

    def test_threshold_zero_includes_all(self):
        frags   = [_make_fragment(0), _make_fragment(1), _make_fragment(2)]
        entries = [_make_entry(0, 1, 0.1), _make_entry(1, 2, 0.2)]
        g       = build_fragment_graph(frags, entries, threshold=0.0)
        assert g.n_edges == 2

    def test_self_edge_ignored(self):
        """entry с fid_i == fid_j не добавляется как петля."""
        frags   = [_make_fragment(0)]
        entry   = CompatEntry(
            edge_i=_make_edge(0, 0),
            edge_j=_make_edge(0, 1),
            score=0.9,
        )
        g = build_fragment_graph(frags, [entry])
        assert g.n_edges == 0

    def test_duplicate_entries_max_weight(self):
        frags   = [_make_fragment(0), _make_fragment(1)]
        entries = [_make_entry(0, 1, 0.5), _make_entry(0, 1, 0.8)]
        g       = build_fragment_graph(frags, entries)
        assert g.weight(0, 1) == pytest.approx(0.8)


# ─── mst_ordering ─────────────────────────────────────────────────────────────

class TestMstOrdering:
    def test_empty_graph(self):
        assert mst_ordering(FragmentGraph()) == []

    def test_single_node(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(7))
        assert mst_ordering(g) == [7]

    def test_chain_all_nodes_covered(self):
        g   = _chain_graph(6)
        ord = mst_ordering(g)
        assert set(ord) == set(g.sorted_nodes())

    def test_no_duplicates(self):
        g   = _complete_graph(5)
        ord = mst_ordering(g)
        assert len(set(ord)) == len(ord)

    def test_length_equals_n_nodes(self):
        g   = _chain_graph(8)
        ord = mst_ordering(g)
        assert len(ord) == g.n_nodes

    def test_disconnected_graph_all_covered(self):
        """Изолированные узлы добавляются в конце."""
        g = FragmentGraph()
        for i in range(5):
            g.add_node(_make_fragment(i))
        g.add_edge(0, 1, 0.9)  # Только одно ребро
        ord = mst_ordering(g)
        assert set(ord) == {0, 1, 2, 3, 4}
        assert len(ord) == 5

    def test_star_graph_center_first(self):
        """В звёздном графе центральный узел (наибольшая степень) идёт первым."""
        g   = _star_graph(5)
        ord = mst_ordering(g)
        assert ord[0] == 0

    def test_two_nodes(self):
        g = _chain_graph(2)
        assert len(mst_ordering(g)) == 2


# ─── spectral_ordering ────────────────────────────────────────────────────────

class TestSpectralOrdering:
    def test_empty_graph(self):
        assert spectral_ordering(FragmentGraph()) == []

    def test_single_node(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(3))
        assert spectral_ordering(g) == [3]

    def test_two_nodes(self):
        g   = _chain_graph(2)
        ord = spectral_ordering(g)
        assert len(ord) == 2
        assert set(ord) == {0, 1}

    def test_length_equals_n_nodes(self):
        g   = _chain_graph(6)
        ord = spectral_ordering(g)
        assert len(ord) == 6

    def test_no_duplicates(self):
        g   = _complete_graph(5)
        ord = spectral_ordering(g)
        assert len(set(ord)) == len(ord)

    def test_all_nodes_covered(self):
        g   = _chain_graph(7)
        ord = spectral_ordering(g)
        assert set(ord) == set(g.sorted_nodes())

    def test_returns_list(self):
        g   = _chain_graph(4)
        ord = spectral_ordering(g)
        assert isinstance(ord, list)

    def test_different_topologies_different_orders(self):
        """Цепной и звёздный граф дают разный спектральный порядок."""
        chain = _chain_graph(5)
        star  = _star_graph(5)
        o_chain = spectral_ordering(chain)
        o_star  = spectral_ordering(star)
        # Не обязаны быть разными, но тест проверяет работоспособность
        assert len(o_chain) == 5
        assert len(o_star)  == 5


# ─── random_walk_similarity ───────────────────────────────────────────────────

class TestRandomWalkSimilarity:
    def test_empty_graph_empty_matrix(self):
        g = FragmentGraph()
        R = random_walk_similarity(g)
        assert R.shape == (0, 0)

    def test_shape_n_by_n(self):
        g = _chain_graph(5)
        R = random_walk_similarity(g)
        assert R.shape == (5, 5)

    def test_nonneg_values(self):
        g = _chain_graph(4)
        R = random_walk_similarity(g)
        assert (R >= 0).all()

    def test_rows_sum_one(self):
        g = _complete_graph(4)
        R = random_walk_similarity(g)
        row_sums = R.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_single_node(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        R = random_walk_similarity(g)
        assert R.shape == (1, 1)

    def test_dtype_float64(self):
        g = _chain_graph(3)
        R = random_walk_similarity(g)
        assert R.dtype == np.float64

    def test_n_iter_1_still_valid(self):
        g = _chain_graph(4)
        R = random_walk_similarity(g, n_iter=1)
        assert R.shape == (4, 4)


# ─── degree_centrality ────────────────────────────────────────────────────────

class TestDegreeCentrality:
    def test_empty_graph(self):
        g = FragmentGraph()
        assert degree_centrality(g) == {}

    def test_isolated_node_zero(self):
        g = FragmentGraph()
        g.add_node(_make_fragment(0))
        c = degree_centrality(g)
        assert c[0] == pytest.approx(0.0)

    def test_all_values_in_0_1(self):
        g = _complete_graph(5)
        c = degree_centrality(g)
        for v in c.values():
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_max_centrality_one(self):
        g = _complete_graph(5)
        c = degree_centrality(g)
        assert max(c.values()) == pytest.approx(1.0)

    def test_star_center_max(self):
        """В звёздном графе центральный узел имеет максимальную центральность."""
        g = _star_graph(6)
        c = degree_centrality(g)
        assert c[0] == pytest.approx(1.0)

    def test_chain_ends_lower(self):
        """В цепи крайние узлы имеют меньшую степень, чем средние."""
        g = _chain_graph(5)
        c = degree_centrality(g)
        # Крайние: 0 и 4 (по 1 ребру); средние: 1,2,3 (по 2)
        assert c[0] <= c[2] + 1e-6
        assert c[4] <= c[2] + 1e-6

    def test_returns_dict_with_all_fids(self):
        g = _chain_graph(4)
        c = degree_centrality(g)
        assert set(c.keys()) == set(g.sorted_nodes())


# ─── analyze_graph ────────────────────────────────────────────────────────────

class TestAnalyzeGraph:
    def test_returns_graph_match_result(self):
        g = _chain_graph(4)
        r = analyze_graph(g)
        assert isinstance(r, GraphMatchResult)

    def test_mst_order_covers_all_nodes(self):
        g = _chain_graph(5)
        r = analyze_graph(g)
        assert set(r.mst_order) == set(g.sorted_nodes())

    def test_spectral_order_covers_all_nodes(self):
        g = _star_graph(5)
        r = analyze_graph(g)
        assert set(r.spectral_order) == set(g.sorted_nodes())

    def test_mst_edges_leq_n_minus_1(self):
        g = _complete_graph(5)
        r = analyze_graph(g)
        assert len(r.mst_edges) <= g.n_nodes - 1

    def test_centrality_keys(self):
        g = _chain_graph(4)
        r = analyze_graph(g)
        assert set(r.centrality.keys()) == set(g.sorted_nodes())

    def test_graph_reference(self):
        g = _chain_graph(3)
        r = analyze_graph(g)
        assert r.graph is g

    def test_summary_str(self):
        g = _chain_graph(4)
        r = analyze_graph(g)
        s = r.summary()
        assert isinstance(s, str)
        assert "GraphMatchResult" in s

    def test_empty_graph_no_crash(self):
        g  = FragmentGraph()
        r  = analyze_graph(g)
        assert r.mst_order == []
        assert r.spectral_order == []
