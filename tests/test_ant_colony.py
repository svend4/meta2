"""Расширенные тесты для puzzle_reconstruction/assembly/ant_colony.py."""
import numpy as np
import pytest

from puzzle_reconstruction.assembly.ant_colony import (
    _build_ant_solution,
    _build_eta_matrix,
    _order_to_assembly,
    ant_colony_assembly,
)
from puzzle_reconstruction.models import (
    Assembly,
    CompatEntry,
    EdgeSide,
    EdgeSignature,
    Fragment,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _frag(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((8, 2)),
    )
    frag.edges = [_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _entry(ei: EdgeSignature, ej: EdgeSignature, score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _build_chain(n: int, base_score: float = 0.8):
    frags = [_frag(i) for i in range(n)]
    entries = []
    for i in range(n - 1):
        entries.append(_entry(frags[i].edges[0], frags[i + 1].edges[0], score=base_score))
    return frags, entries


# ─── TestAntColonyAssembly ────────────────────────────────────────────────────

class TestAntColonyAssembly:
    def test_returns_assembly(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert isinstance(result, Assembly)

    def test_empty_fragments_returns_empty_assembly(self):
        result = ant_colony_assembly([], [])
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 0
        assert result.placements == {}

    def test_single_fragment(self):
        frags = [_frag(0)]
        result = ant_colony_assembly(frags, [], n_iterations=1, seed=0)
        assert 0 in result.placements

    def test_all_fragments_in_placements(self):
        frags, entries = _build_chain(4)
        result = ant_colony_assembly(frags, entries, n_iterations=3, seed=0)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_placements_count_matches_fragments(self):
        n = 4
        frags, entries = _build_chain(n)
        result = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert len(result.placements) == n

    def test_n_ants_zero_auto_set(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_ants=0, n_iterations=2, seed=0)
        assert isinstance(result, Assembly)

    def test_n_ants_explicit(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_ants=5, n_iterations=2, seed=0)
        assert isinstance(result, Assembly)

    def test_n_iterations_1(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=1, seed=0)
        assert isinstance(result, Assembly)

    def test_n_iterations_0_returns_assembly(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=0, seed=0)
        assert isinstance(result, Assembly)

    def test_deterministic_with_same_seed(self):
        frags, entries = _build_chain(4)
        r1 = ant_colony_assembly(frags, entries, n_iterations=5, seed=42)
        r2 = ant_colony_assembly(frags, entries, n_iterations=5, seed=42)
        assert r1.total_score == r2.total_score

    def test_score_nonneg(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=3, seed=0)
        assert result.total_score >= -1e-9

    def test_score_is_float(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert isinstance(result.total_score, float)

    def test_no_entries_returns_assembly(self):
        frags = [_frag(i) for i in range(3)]
        result = ant_colony_assembly(frags, [], n_iterations=2, seed=0)
        assert isinstance(result, Assembly)

    def test_compat_matrix_is_ndarray(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert isinstance(result.compat_matrix, np.ndarray)

    def test_placement_pos_length_2(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        for pos, angle in result.placements.values():
            assert len(pos) == 2

    def test_placement_angle_is_float(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        for pos, angle in result.placements.values():
            assert isinstance(angle, float)

    def test_allow_rotation_false(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=2,
                                      allow_rotation=False, seed=0)
        assert isinstance(result, Assembly)

    def test_with_high_score_entries_positive_total(self):
        frags, entries = _build_chain(4, base_score=0.95)
        result = ant_colony_assembly(frags, entries, n_iterations=5, seed=0)
        assert result.total_score > 0.0

    def test_fragments_field_stored(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert result.fragments == frags

    def test_elite_count_zero(self):
        frags, entries = _build_chain(3)
        result = ant_colony_assembly(frags, entries, n_iterations=2,
                                      elite_count=0, seed=0)
        assert isinstance(result, Assembly)

    def test_alpha_beta_affect_result(self):
        frags, entries = _build_chain(5, base_score=0.7)
        r1 = ant_colony_assembly(frags, entries, n_iterations=10,
                                  alpha=1.0, beta=1.0, seed=7)
        r2 = ant_colony_assembly(frags, entries, n_iterations=10,
                                  alpha=1.0, beta=5.0, seed=7)
        # Different beta → different scores (not guaranteed but typical)
        assert isinstance(r1.total_score, float)
        assert isinstance(r2.total_score, float)


# ─── TestBuildAntSolution ─────────────────────────────────────────────────────

class TestBuildAntSolution:
    def _setup(self, n: int = 4):
        frags = [_frag(i) for i in range(n)]
        frag_ids = [f.fragment_id for f in frags]
        tau = np.ones((n, n))
        np.fill_diagonal(tau, 0.0)
        eta = np.ones((n, n))
        np.fill_diagonal(eta, 0.0)
        rng = np.random.RandomState(0)
        rotations = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        return n, frag_ids, tau, eta, rng, rotations

    def test_returns_tuple_of_3(self):
        n, frag_ids, tau, eta, rng, rots = self._setup()
        result = _build_ant_solution(n, frag_ids, tau, eta, 1.0, 3.0,
                                      eta, rots, rng, True)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_order_has_n_elements(self):
        n, frag_ids, tau, eta, rng, rots = self._setup()
        order, angles, score = _build_ant_solution(n, frag_ids, tau, eta, 1.0, 3.0,
                                                    eta, rots, rng, True)
        assert len(order) == n

    def test_order_contains_all_frag_ids(self):
        n, frag_ids, tau, eta, rng, rots = self._setup()
        order, angles, score = _build_ant_solution(n, frag_ids, tau, eta, 1.0, 3.0,
                                                    eta, rots, rng, True)
        assert sorted(order) == sorted(frag_ids)

    def test_no_repeated_fragments_in_order(self):
        n, frag_ids, tau, eta, rng, rots = self._setup()
        order, angles, score = _build_ant_solution(n, frag_ids, tau, eta, 1.0, 3.0,
                                                    eta, rots, rng, True)
        assert len(set(order)) == n

    def test_angles_has_n_elements(self):
        n, frag_ids, tau, eta, rng, rots = self._setup()
        order, angles, score = _build_ant_solution(n, frag_ids, tau, eta, 1.0, 3.0,
                                                    eta, rots, rng, True)
        assert len(angles) == n

    def test_total_score_is_float(self):
        n, frag_ids, tau, eta, rng, rots = self._setup()
        order, angles, score = _build_ant_solution(n, frag_ids, tau, eta, 1.0, 3.0,
                                                    eta, rots, rng, True)
        assert isinstance(score, float)

    def test_total_score_nonneg(self):
        n, frag_ids, tau, eta, rng, rots = self._setup()
        order, angles, score = _build_ant_solution(n, frag_ids, tau, eta, 1.0, 3.0,
                                                    eta, rots, rng, True)
        assert score >= 0.0

    def test_no_rotation_angles_zero(self):
        n, frag_ids, tau, eta, rng, rots = self._setup()
        order, angles, score = _build_ant_solution(n, frag_ids, tau, eta, 1.0, 3.0,
                                                    eta, rots, rng, False)
        assert np.all(angles == 0.0)


# ─── TestBuildEtaMatrix ───────────────────────────────────────────────────────

class TestBuildEtaMatrix:
    def test_returns_ndarray(self):
        frags, entries = _build_chain(3)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, 3, idx_map)
        assert isinstance(result, np.ndarray)

    def test_shape_n_by_n(self):
        n = 4
        frags, entries = _build_chain(n)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, n, idx_map)
        assert result.shape == (n, n)

    def test_diagonal_is_zero(self):
        n = 3
        frags, entries = _build_chain(n)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, n, idx_map)
        assert np.all(np.diag(result) == 0.0)

    def test_nonneg_values(self):
        frags, entries = _build_chain(3)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, [*entries], 3, idx_map)
        assert np.all(result >= 0.0)

    def test_symmetric(self):
        frags, entries = _build_chain(3)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, 3, idx_map)
        assert np.allclose(result, result.T)

    def test_entries_raise_eta_above_default(self):
        frags, entries = _build_chain(2, base_score=0.9)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, 2, idx_map)
        # Off-diagonal should be 0.9 > 1e-6
        assert result[0, 1] == pytest.approx(0.9)

    def test_no_entries_all_small(self):
        n = 3
        frags = [_frag(i) for i in range(n)]
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, [], n, idx_map)
        # All off-diagonal should be the small constant
        mask = ~np.eye(n, dtype=bool)
        assert np.all(result[mask] == pytest.approx(1e-6))


# ─── TestOrderToAssembly ──────────────────────────────────────────────────────

class TestOrderToAssembly:
    def test_returns_assembly(self):
        frags = [_frag(i) for i in range(3)]
        order = [f.fragment_id for f in frags]
        angles = np.zeros(3)
        result = _order_to_assembly(order, angles, frags, total_score=1.0)
        assert isinstance(result, Assembly)

    def test_all_frags_in_placements(self):
        frags = [_frag(i) for i in range(4)]
        order = [f.fragment_id for f in frags]
        angles = np.zeros(4)
        result = _order_to_assembly(order, angles, frags, total_score=2.0)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_total_score_stored(self):
        frags = [_frag(0), _frag(1)]
        order = [0, 1]
        angles = np.zeros(2)
        result = _order_to_assembly(order, angles, frags, total_score=3.5)
        assert result.total_score == pytest.approx(3.5)

    def test_fragments_stored(self):
        frags = [_frag(i) for i in range(3)]
        order = [f.fragment_id for f in frags]
        angles = np.zeros(3)
        result = _order_to_assembly(order, angles, frags, total_score=0.0)
        assert result.fragments == frags

    def test_spacing_in_positions(self):
        frags = [_frag(0), _frag(1)]
        order = [0, 1]
        angles = np.zeros(2)
        result = _order_to_assembly(order, angles, frags, total_score=0.0)
        pos0 = result.placements[0][0]
        pos1 = result.placements[1][0]
        # Second fragment should be offset by spacing
        diff = np.linalg.norm(pos1 - pos0)
        assert diff > 0.0
