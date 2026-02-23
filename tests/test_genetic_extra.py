"""Extra tests for puzzle_reconstruction.assembly.genetic."""
import numpy as np
import pytest

from puzzle_reconstruction.assembly.genetic import (
    _build_score_map,
    _fitness,
    _individual_to_assembly,
    _make_individual,
    _mutate,
    _order_crossover,
    _tournament_select,
    genetic_assembly,
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


def _make_etf(frags):
    return {e.edge_id: f.fragment_id for f in frags for e in f.edges}


# ─── TestGeneticAssemblyExtra ───────────────────────────────────────────────

class TestGeneticAssemblyExtra:
    def test_returns_assembly_type(self):
        frags, entries = _build_chain(3)
        r = genetic_assembly(frags, entries, n_generations=1, population_size=4)
        assert isinstance(r, Assembly)

    def test_all_frags_in_placements(self):
        frags, entries = _build_chain(4)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        for f in frags:
            assert f.fragment_id in r.placements

    def test_placements_count(self):
        frags, entries = _build_chain(5)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=5)
        assert len(r.placements) == 5

    def test_score_nonneg(self):
        frags, entries = _build_chain(3)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert r.total_score >= 0.0

    def test_score_is_float(self):
        frags, entries = _build_chain(3)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert isinstance(r.total_score, float)

    def test_empty_returns_assembly(self):
        r = genetic_assembly([], [])
        assert isinstance(r, Assembly)
        assert len(r.placements) == 0

    def test_single_frag(self):
        frags = [_frag(0)]
        r = genetic_assembly(frags, [], n_generations=1, population_size=2)
        assert 0 in r.placements

    def test_seed_reproducibility(self):
        frags, entries = _build_chain(4)
        r1 = genetic_assembly(frags, entries, n_generations=3, population_size=4, seed=7)
        r2 = genetic_assembly(frags, entries, n_generations=3, population_size=4, seed=7)
        assert r1.total_score == pytest.approx(r2.total_score)

    def test_fragments_reference(self):
        frags, entries = _build_chain(3)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert r.fragments is frags

    def test_compat_matrix_ndarray(self):
        frags, entries = _build_chain(3)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert isinstance(r.compat_matrix, np.ndarray)

    def test_allow_rotation_false(self):
        frags, entries = _build_chain(3)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=4,
                              allow_rotation=False)
        assert isinstance(r, Assembly)

    def test_no_entries_zero_score(self):
        frags = [_frag(i) for i in range(3)]
        r = genetic_assembly(frags, [], n_generations=2, population_size=4)
        assert r.total_score == pytest.approx(0.0)

    def test_high_score_entries_positive(self):
        frags, entries = _build_chain(4, base_score=0.95)
        r = genetic_assembly(frags, entries, n_generations=3, population_size=5)
        assert r.total_score > 0.0

    def test_pos_length_2(self):
        frags, entries = _build_chain(3)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        for pos, angle in r.placements.values():
            assert len(pos) == 2

    def test_angle_is_float(self):
        frags, entries = _build_chain(3)
        r = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        for pos, angle in r.placements.values():
            assert isinstance(angle, float)


# ─── TestMakeIndividualExtra ─────────────────────────────────────────────────

class TestMakeIndividualExtra:
    def _call(self, n=4, seed=0):
        fids = np.arange(n)
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        rng = np.random.RandomState(seed)
        return _make_individual(fids, rots, rng)

    def test_returns_tuple_len_2(self):
        ind = self._call()
        assert isinstance(ind, tuple) and len(ind) == 2

    def test_order_is_permutation(self):
        order, _ = self._call(n=5)
        assert sorted(order.tolist()) == list(range(5))

    def test_angles_are_valid_rots(self):
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        fids = np.arange(4)
        _, angles = _make_individual(fids, rots, np.random.RandomState(1))
        for a in angles:
            assert any(abs(a - r) < 1e-9 for r in rots)

    def test_different_seeds_different_orders(self):
        order1, _ = self._call(n=6, seed=0)
        order2, _ = self._call(n=6, seed=99)
        assert not np.array_equal(order1, order2)


# ─── TestFitnessExtra ────────────────────────────────────────────────────────

class TestFitnessExtra:
    def test_zero_with_empty_map(self):
        fids = np.array([0, 1, 2])
        ind = (fids.copy(), np.zeros(3))
        assert _fitness(ind, {}, fids) == pytest.approx(0.0)

    def test_correct_sum(self):
        fids = np.array([0, 1, 2])
        ind = (np.array([0, 1, 2]), np.zeros(3))
        sm = {(0, 1): 0.5, (1, 2): 0.7}
        assert _fitness(ind, sm, fids) == pytest.approx(1.2)

    def test_missing_pair_contributes_zero(self):
        fids = np.array([0, 1, 2])
        ind = (np.array([0, 1, 2]), np.zeros(3))
        sm = {(0, 1): 0.5}
        result = _fitness(ind, sm, fids)
        assert result == pytest.approx(0.5)

    def test_nonneg(self):
        fids = np.array([0, 1, 2])
        ind = (fids.copy(), np.zeros(3))
        assert _fitness(ind, {(0, 1): 0.3, (1, 2): 0.4}, fids) >= 0.0

    def test_returns_float(self):
        fids = np.array([0, 1])
        ind = (fids.copy(), np.zeros(2))
        assert isinstance(_fitness(ind, {}, fids), float)


# ─── TestOrderCrossoverExtra ─────────────────────────────────────────────────

class TestOrderCrossoverExtra:
    def _parent(self, n, seed):
        return (np.random.RandomState(seed).permutation(np.arange(n)),
                np.zeros(n))

    def test_child_is_permutation(self):
        n = 6
        order, _ = _order_crossover(self._parent(n, 0), self._parent(n, 1),
                                     np.random.RandomState(2))
        assert sorted(order.tolist()) == list(range(n))

    def test_length_preserved(self):
        n = 7
        order, _ = _order_crossover(self._parent(n, 3), self._parent(n, 4),
                                     np.random.RandomState(5))
        assert len(order) == n

    def test_single_element(self):
        p = (np.array([0]), np.zeros(1))
        order, _ = _order_crossover(p, p, np.random.RandomState(0))
        assert order[0] == 0

    def test_two_elements(self):
        p1 = (np.array([0, 1]), np.zeros(2))
        p2 = (np.array([1, 0]), np.zeros(2))
        order, _ = _order_crossover(p1, p2, np.random.RandomState(0))
        assert sorted(order.tolist()) == [0, 1]


# ─── TestTournamentSelectExtra ───────────────────────────────────────────────

class TestTournamentSelectExtra:
    def test_returns_tuple(self):
        pop = [(np.array([0, 1]), np.zeros(2)),
               (np.array([1, 0]), np.zeros(2))]
        scores = np.array([0.3, 0.8])
        result = _tournament_select(pop, scores, k=2,
                                    rng=np.random.RandomState(0))
        assert isinstance(result, tuple)

    def test_result_in_population(self):
        pop = [(np.array([i]), np.zeros(1)) for i in range(4)]
        scores = np.array([0.1, 0.5, 0.3, 0.9])
        result = _tournament_select(pop, scores, k=3,
                                    rng=np.random.RandomState(0))
        assert any(np.array_equal(result[0], p[0]) for p in pop)

    def test_k_1(self):
        pop = [(np.array([0, 1]), np.zeros(2))]
        scores = np.array([0.5])
        result = _tournament_select(pop, scores, k=1,
                                    rng=np.random.RandomState(0))
        assert isinstance(result, tuple)

    def test_selects_best_with_large_k(self):
        pop = [(np.array([i]), np.zeros(1)) for i in range(5)]
        scores = np.array([0.1, 0.9, 0.2, 0.3, 0.4])
        result = _tournament_select(pop, scores, k=5,
                                    rng=np.random.RandomState(0))
        assert result[0][0] == 1   # highest score


# ─── TestMutateExtra ─────────────────────────────────────────────────────────

class TestMutateExtra:
    def test_permutation_preserved(self):
        n = 6
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        order = np.arange(n)
        result_order, _ = _mutate((order.copy(), np.zeros(n)), rots,
                                   np.random.RandomState(0))
        assert sorted(result_order.tolist()) == list(range(n))

    def test_length_preserved(self):
        n = 5
        rots = np.array([0.0, np.pi / 2])
        order = np.arange(n)
        r_order, r_angles = _mutate((order, np.zeros(n)), rots,
                                     np.random.RandomState(1))
        assert len(r_order) == n
        assert len(r_angles) == n

    def test_single_element(self):
        result, _ = _mutate((np.array([7]), np.zeros(1)),
                             np.array([0.0]), np.random.RandomState(0))
        assert result[0] == 7

    def test_angles_are_valid_rotations(self):
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        order = np.arange(5)
        _, angles = _mutate((order.copy(), np.zeros(5)), rots,
                             np.random.RandomState(2))
        for a in angles:
            assert any(abs(a - r) < 1e-9 for r in rots)


# ─── TestBuildScoreMapExtra ──────────────────────────────────────────────────

class TestBuildScoreMapExtra:
    def test_returns_dict(self):
        frags, entries = _build_chain(3)
        etf = _make_etf(frags)
        assert isinstance(_build_score_map(entries, etf), dict)

    def test_keys_sorted_pairs(self):
        frags, entries = _build_chain(4)
        etf = _make_etf(frags)
        for a, b in _build_score_map(entries, etf).keys():
            assert a <= b

    def test_values_are_floats(self):
        frags, entries = _build_chain(3)
        etf = _make_etf(frags)
        for v in _build_score_map(entries, etf).values():
            assert isinstance(v, float)

    def test_empty_entries_empty_map(self):
        assert _build_score_map([], {}) == {}

    def test_same_frag_skipped(self):
        f0 = _frag(0)
        e = _entry(f0.edges[0], f0.edges[1], score=0.9)
        etf = {ed.edge_id: f0.fragment_id for ed in f0.edges}
        assert _build_score_map([e], etf) == {}

    def test_max_score_kept(self):
        f0, f1 = _frag(0), _frag(1)
        etf = _make_etf([f0, f1])
        e_low = _entry(f0.edges[0], f1.edges[0], score=0.2)
        e_high = _entry(f0.edges[0], f1.edges[0], score=0.9)
        result = _build_score_map([e_low, e_high], etf)
        assert list(result.values())[0] == pytest.approx(0.9)
