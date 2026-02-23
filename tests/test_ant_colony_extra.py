"""Extra tests for puzzle_reconstruction.assembly.ant_colony."""
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _edge(edge_id):
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _frag(fid, n_edges=2):
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        mask=np.zeros((32, 32), dtype=np.uint8),
        contour=np.zeros((8, 2)),
    )
    frag.edges = [_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _entry(ei, ej, score=0.5):
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _build_chain(n, base_score=0.8):
    frags = [_frag(i) for i in range(n)]
    entries = [
        _entry(frags[i].edges[0], frags[i + 1].edges[0], score=base_score)
        for i in range(n - 1)
    ]
    return frags, entries


# ─── TestAntColonyAssemblyExtra ──────────────────────────────────────────────

class TestAntColonyAssemblyExtra:
    def test_returns_assembly(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert isinstance(r, Assembly)

    def test_all_frags_in_placements(self):
        frags, entries = _build_chain(4)
        r = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        for f in frags:
            assert f.fragment_id in r.placements

    def test_placements_count(self):
        n = 5
        frags, entries = _build_chain(n)
        r = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert len(r.placements) == n

    def test_score_nonneg(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=3, seed=0)
        assert r.total_score >= 0.0

    def test_score_is_float(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert isinstance(r.total_score, float)

    def test_empty_returns_assembly(self):
        r = ant_colony_assembly([], [])
        assert isinstance(r, Assembly)
        assert len(r.placements) == 0

    def test_single_frag(self):
        r = ant_colony_assembly([_frag(0)], [], n_iterations=1, seed=0)
        assert 0 in r.placements

    def test_seed_reproducibility(self):
        frags, entries = _build_chain(4)
        r1 = ant_colony_assembly(frags, entries, n_iterations=5, seed=42)
        r2 = ant_colony_assembly(frags, entries, n_iterations=5, seed=42)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_fragments_field(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert r.fragments == frags

    def test_compat_matrix_ndarray(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        assert isinstance(r.compat_matrix, np.ndarray)

    def test_pos_length_2(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        for pos, angle in r.placements.values():
            assert len(pos) == 2

    def test_angle_is_float(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=2, seed=0)
        for pos, angle in r.placements.values():
            assert isinstance(angle, float)

    def test_allow_rotation_false(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=2,
                                 allow_rotation=False, seed=0)
        assert isinstance(r, Assembly)

    def test_no_entries_score_nonneg(self):
        frags = [_frag(i) for i in range(3)]
        r = ant_colony_assembly(frags, [], n_iterations=2, seed=0)
        assert r.total_score >= 0.0

    def test_n_ants_explicit(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_ants=3, n_iterations=2, seed=0)
        assert isinstance(r, Assembly)

    def test_elite_count_zero(self):
        frags, entries = _build_chain(3)
        r = ant_colony_assembly(frags, entries, n_iterations=2,
                                 elite_count=0, seed=0)
        assert isinstance(r, Assembly)

    def test_high_score_positive(self):
        frags, entries = _build_chain(4, base_score=0.95)
        r = ant_colony_assembly(frags, entries, n_iterations=5, seed=0)
        assert r.total_score > 0.0


# ─── TestBuildAntSolutionExtra ───────────────────────────────────────────────

class TestBuildAntSolutionExtra:
    def _setup(self, n=4):
        fids = list(range(n))
        tau = np.ones((n, n))
        np.fill_diagonal(tau, 0.0)
        eta = np.ones((n, n))
        np.fill_diagonal(eta, 0.0)
        rng = np.random.RandomState(0)
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        return n, fids, tau, eta, rng, rots

    def test_returns_3_tuple(self):
        n, fids, tau, eta, rng, rots = self._setup()
        result = _build_ant_solution(n, fids, tau, eta, 1.0, 2.0,
                                      eta, rots, rng, True)
        assert len(result) == 3

    def test_order_len_n(self):
        n, fids, tau, eta, rng, rots = self._setup()
        order, angles, score = _build_ant_solution(n, fids, tau, eta,
                                                    1.0, 2.0, eta, rots, rng, True)
        assert len(order) == n

    def test_order_is_permutation(self):
        n, fids, tau, eta, rng, rots = self._setup()
        order, _, _ = _build_ant_solution(n, fids, tau, eta,
                                           1.0, 2.0, eta, rots, rng, True)
        assert sorted(order) == sorted(fids)

    def test_no_repeated(self):
        n, fids, tau, eta, rng, rots = self._setup()
        order, _, _ = _build_ant_solution(n, fids, tau, eta,
                                           1.0, 2.0, eta, rots, rng, True)
        assert len(set(order)) == n

    def test_angles_len_n(self):
        n, fids, tau, eta, rng, rots = self._setup()
        _, angles, _ = _build_ant_solution(n, fids, tau, eta,
                                            1.0, 2.0, eta, rots, rng, True)
        assert len(angles) == n

    def test_score_is_float(self):
        n, fids, tau, eta, rng, rots = self._setup()
        _, _, score = _build_ant_solution(n, fids, tau, eta,
                                           1.0, 2.0, eta, rots, rng, True)
        assert isinstance(score, float)

    def test_score_nonneg(self):
        n, fids, tau, eta, rng, rots = self._setup()
        _, _, score = _build_ant_solution(n, fids, tau, eta,
                                           1.0, 2.0, eta, rots, rng, True)
        assert score >= 0.0

    def test_no_rotation_zero_angles(self):
        n, fids, tau, eta, rng, rots = self._setup()
        _, angles, _ = _build_ant_solution(n, fids, tau, eta,
                                            1.0, 2.0, eta, rots, rng, False)
        assert np.all(angles == 0.0)


# ─── TestBuildEtaMatrixExtra ─────────────────────────────────────────────────

class TestBuildEtaMatrixExtra:
    def test_shape_n_n(self):
        n = 5
        frags, entries = _build_chain(n)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, n, idx_map)
        assert result.shape == (n, n)

    def test_diagonal_zero(self):
        n = 4
        frags, entries = _build_chain(n)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, n, idx_map)
        assert np.all(np.diag(result) == 0.0)

    def test_all_nonneg(self):
        frags, entries = _build_chain(3)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, 3, idx_map)
        assert np.all(result >= 0.0)

    def test_symmetric(self):
        frags, entries = _build_chain(4)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, 4, idx_map)
        assert np.allclose(result, result.T)

    def test_off_diagonal_equals_score(self):
        frags, entries = _build_chain(2, base_score=0.75)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, entries, 2, idx_map)
        assert result[0, 1] == pytest.approx(0.75)

    def test_no_entries_small_value(self):
        n = 3
        frags = [_frag(i) for i in range(n)]
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        result = _build_eta_matrix(frags, [], n, idx_map)
        mask = ~np.eye(n, dtype=bool)
        assert np.all(result[mask] == pytest.approx(1e-6))

    def test_returns_ndarray(self):
        frags, entries = _build_chain(3)
        idx_map = {f.fragment_id: i for i, f in enumerate(frags)}
        assert isinstance(_build_eta_matrix(frags, entries, 3, idx_map), np.ndarray)


# ─── TestOrderToAssemblyExtra ────────────────────────────────────────────────

class TestOrderToAssemblyExtra:
    def test_returns_assembly(self):
        frags = [_frag(i) for i in range(3)]
        order = [f.fragment_id for f in frags]
        r = _order_to_assembly(order, np.zeros(3), frags, total_score=1.0)
        assert isinstance(r, Assembly)

    def test_all_frags_in_placements(self):
        frags = [_frag(i) for i in range(4)]
        order = [f.fragment_id for f in frags]
        r = _order_to_assembly(order, np.zeros(4), frags, total_score=2.0)
        for f in frags:
            assert f.fragment_id in r.placements

    def test_total_score_stored(self):
        frags = [_frag(0), _frag(1)]
        r = _order_to_assembly([0, 1], np.zeros(2), frags, total_score=3.5)
        assert r.total_score == pytest.approx(3.5)

    def test_fragments_stored(self):
        frags = [_frag(i) for i in range(3)]
        order = [f.fragment_id for f in frags]
        r = _order_to_assembly(order, np.zeros(3), frags, total_score=0.0)
        assert r.fragments == frags

    def test_spacing_positive(self):
        frags = [_frag(0), _frag(1), _frag(2)]
        order = [0, 1, 2]
        r = _order_to_assembly(order, np.zeros(3), frags, total_score=0.0)
        pos0 = r.placements[0][0]
        pos1 = r.placements[1][0]
        assert np.linalg.norm(pos1 - pos0) > 0.0

    def test_single_frag(self):
        frags = [_frag(42)]
        r = _order_to_assembly([42], np.zeros(1), frags, total_score=0.0)
        assert 42 in r.placements
