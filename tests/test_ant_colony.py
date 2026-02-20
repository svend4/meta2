"""
Тесты для puzzle_reconstruction/assembly/ant_colony.py

Покрытие:
    ant_colony_assembly   — пустой, 1 фрагмент, базовый N=4,
                            все фрагменты размещены, структура placements,
                            score ≥ 0, воспроизводимость
    _build_eta_matrix     — нулевые entries, симметричность, диагональ = 0
    _order_to_assembly    — корректная структура Assembly
    _build_ant_solution   — корректная перестановка, score ≥ 0
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide,
)
from puzzle_reconstruction.assembly.ant_colony import (
    ant_colony_assembly,
    _build_eta_matrix,
    _order_to_assembly,
    _build_ant_solution,
)


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int, length: float = 80.0) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=length,
    )


def _make_fragment(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((8, 2)),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _make_entry(ei: EdgeSignature, ej: EdgeSignature,
                score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


@pytest.fixture
def frags4():
    return [_make_fragment(i) for i in range(4)]


@pytest.fixture
def entries4(frags4):
    entries = []
    for i, fi in enumerate(frags4):
        for j, fj in enumerate(frags4):
            if i >= j:
                continue
            s = 0.1 * (i + j + 1)
            entries.append(_make_entry(fi.edges[0], fj.edges[0], min(s, 1.0)))
    return entries


# ─── ant_colony_assembly: граничные случаи ────────────────────────────────────

class TestACOEdgeCases:
    def test_empty_fragments(self):
        result = ant_colony_assembly([], [], n_ants=5, n_iterations=2)
        assert isinstance(result, Assembly)
        assert result.fragments == []

    def test_single_fragment(self):
        frag = _make_fragment(0)
        result = ant_colony_assembly([frag], [], n_ants=3, n_iterations=2)
        assert isinstance(result, Assembly)
        assert 0 in result.placements


# ─── ant_colony_assembly: базовые свойства ────────────────────────────────────

class TestACOBasic:
    def test_returns_assembly(self, frags4, entries4):
        result = ant_colony_assembly(
            frags4, entries4, n_ants=5, n_iterations=3, seed=0,
        )
        assert isinstance(result, Assembly)

    def test_all_placed(self, frags4, entries4):
        result = ant_colony_assembly(
            frags4, entries4, n_ants=5, n_iterations=3, seed=0,
        )
        for frag in frags4:
            assert frag.fragment_id in result.placements

    def test_placement_structure(self, frags4, entries4):
        result = ant_colony_assembly(
            frags4, entries4, n_ants=5, n_iterations=3, seed=0,
        )
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (2,)
            assert isinstance(angle, float)

    def test_score_nonneg(self, frags4, entries4):
        result = ant_colony_assembly(
            frags4, entries4, n_ants=5, n_iterations=3, seed=0,
        )
        assert result.total_score >= 0.0

    def test_auto_n_ants(self, frags4, entries4):
        """n_ants=0 → авто-вычисление max(10, N)."""
        result = ant_colony_assembly(
            frags4, entries4, n_ants=0, n_iterations=2, seed=0,
        )
        assert isinstance(result, Assembly)


class TestACOReproducibility:
    def test_same_seed(self, frags4, entries4):
        r1 = ant_colony_assembly(frags4, entries4, n_ants=5, n_iterations=5, seed=7)
        r2 = ant_colony_assembly(frags4, entries4, n_ants=5, n_iterations=5, seed=7)
        assert math.isclose(r1.total_score, r2.total_score)
        for fid in r1.placements:
            p1, a1 = r1.placements[fid]
            p2, a2 = r2.placements[fid]
            assert np.allclose(p1, p2)
            assert math.isclose(a1, a2)

    def test_different_seeds_both_valid(self, frags4, entries4):
        r1 = ant_colony_assembly(frags4, entries4, seed=1)
        r2 = ant_colony_assembly(frags4, entries4, seed=99)
        assert isinstance(r1, Assembly)
        assert isinstance(r2, Assembly)


# ─── _build_eta_matrix ────────────────────────────────────────────────────────

class TestBuildEtaMatrix:
    def test_empty_entries(self, frags4):
        n       = len(frags4)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags4)}
        eta = _build_eta_matrix(frags4, [], n, idx_map)
        assert eta.shape == (n, n)
        np.testing.assert_array_equal(np.diag(eta), 0.0)

    def test_symmetry(self, frags4, entries4):
        n       = len(frags4)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags4)}
        eta = _build_eta_matrix(frags4, entries4, n, idx_map)
        assert np.allclose(eta, eta.T)

    def test_diagonal_zero(self, frags4, entries4):
        n       = len(frags4)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags4)}
        eta = _build_eta_matrix(frags4, entries4, n, idx_map)
        np.testing.assert_array_equal(np.diag(eta), 0.0)

    def test_known_score(self):
        f0, f1 = _make_fragment(0), _make_fragment(1)
        ei, ej = f0.edges[0], f1.edges[0]
        entry  = _make_entry(ei, ej, score=0.8)
        n      = 2
        idx_map = {0: 0, 1: 1}
        eta = _build_eta_matrix([f0, f1], [entry], n, idx_map)
        assert math.isclose(eta[0, 1], 0.8, abs_tol=1e-6)
        assert math.isclose(eta[1, 0], 0.8, abs_tol=1e-6)

    def test_takes_max_score(self):
        """Для одной пары берётся максимальный score."""
        f0, f1 = _make_fragment(0), _make_fragment(1)
        ei, ej = f0.edges[0], f1.edges[0]
        entries = [
            _make_entry(ei, ej, score=0.3),
            _make_entry(ei, ej, score=0.9),
            _make_entry(ei, ej, score=0.6),
        ]
        eta = _build_eta_matrix([f0, f1], entries, 2, {0: 0, 1: 1})
        assert math.isclose(eta[0, 1], 0.9, abs_tol=1e-6)


# ─── _order_to_assembly ───────────────────────────────────────────────────────

class TestOrderToAssembly:
    def test_basic(self, frags4):
        order  = [f.fragment_id for f in frags4]
        angles = np.zeros(len(order))
        result = _order_to_assembly(order, angles, frags4, total_score=1.5)
        assert isinstance(result, Assembly)
        assert result.total_score == 1.5
        for frag in frags4:
            assert frag.fragment_id in result.placements

    def test_grid_layout_distinct_positions(self, frags4):
        order  = [f.fragment_id for f in frags4]
        angles = np.zeros(4)
        result = _order_to_assembly(order, angles, frags4, total_score=0.0)
        positions = [tuple(result.placements[fid][0]) for fid in order]
        assert len(set(positions)) == len(order)

    def test_orphan_fragments_placed(self):
        """Фрагменты не в order должны получить позицию (0,0)."""
        frags  = [_make_fragment(i) for i in range(3)]
        order  = [frags[0].fragment_id]  # Только 1 из 3
        angles = np.zeros(1)
        result = _order_to_assembly(order, angles, frags, total_score=0.0)
        for frag in frags:
            assert frag.fragment_id in result.placements


# ─── _build_ant_solution ──────────────────────────────────────────────────────

class TestBuildAntSolution:
    def _setup(self, n):
        frag_ids = list(range(n))
        tau      = np.ones((n, n))
        np.fill_diagonal(tau, 0.0)
        eta      = tau.copy()
        rotations = np.array([0.0, 1.5707963, 3.1415926, 4.7123889])
        rng      = np.random.RandomState(0)
        return frag_ids, tau, eta, rotations, rng

    def test_permutation_complete(self):
        n = 5
        frag_ids, tau, eta, rotations, rng = self._setup(n)
        order, angles, score = _build_ant_solution(
            n, frag_ids, tau, eta, 1.0, 2.0, eta, rotations, rng, True,
        )
        assert sorted(order) == sorted(frag_ids)

    def test_no_duplicates(self):
        n = 6
        frag_ids, tau, eta, rotations, rng = self._setup(n)
        for _ in range(10):
            order, _, _ = _build_ant_solution(
                n, frag_ids, tau, eta, 1.0, 2.0, eta, rotations, rng, True,
            )
            assert len(set(order)) == n

    def test_score_nonneg(self):
        n = 4
        frag_ids, tau, eta, rotations, rng = self._setup(n)
        _, _, score = _build_ant_solution(
            n, frag_ids, tau, eta, 1.0, 2.0, eta, rotations, rng, False,
        )
        assert score >= 0.0

    def test_angles_shape(self):
        n = 4
        frag_ids, tau, eta, rotations, rng = self._setup(n)
        _, angles, _ = _build_ant_solution(
            n, frag_ids, tau, eta, 1.0, 2.0, eta, rotations, rng, True,
        )
        assert len(angles) == n
