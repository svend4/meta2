"""Расширенные тесты для puzzle_reconstruction/assembly/genetic.py."""
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
    entries = [_entry(frags[i].edges[0], frags[i + 1].edges[0], score=base_score)
               for i in range(n - 1)]
    return frags, entries


def _make_edge_to_frag(frags):
    return {e.edge_id: f.fragment_id for f in frags for e in f.edges}


# ─── TestGeneticAssembly ──────────────────────────────────────────────────────

class TestGeneticAssembly:
    def test_returns_assembly(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert isinstance(result, Assembly)

    def test_empty_fragments_returns_empty(self):
        result = genetic_assembly([], [])
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 0

    def test_single_fragment(self):
        frags = [_frag(0)]
        result = genetic_assembly(frags, [], n_generations=1, population_size=2)
        assert 0 in result.placements

    def test_all_fragments_in_placements(self):
        frags, entries = _build_chain(4)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        for f in frags:
            assert f.fragment_id in result.placements

    def test_score_is_float(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert isinstance(result.total_score, float)

    def test_score_nonneg(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert result.total_score >= 0.0

    def test_deterministic_with_same_seed(self):
        frags, entries = _build_chain(4)
        r1 = genetic_assembly(frags, entries, n_generations=5, population_size=6, seed=0)
        r2 = genetic_assembly(frags, entries, n_generations=5, population_size=6, seed=0)
        assert r1.total_score == r2.total_score

    def test_no_entries_score_zero(self):
        frags = [_frag(i) for i in range(3)]
        result = genetic_assembly(frags, [], n_generations=2, population_size=4)
        assert result.total_score == pytest.approx(0.0)

    def test_with_entries_positive_score(self):
        frags, entries = _build_chain(3, base_score=0.9)
        result = genetic_assembly(frags, entries, n_generations=3, population_size=5)
        assert result.total_score > 0.0

    def test_compat_matrix_is_ndarray(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert isinstance(result.compat_matrix, np.ndarray)

    def test_fragments_stored(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        assert result.fragments is frags

    def test_placement_pos_length_2(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        for pos, angle in result.placements.values():
            assert len(pos) == 2

    def test_placement_angle_is_float(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4)
        for pos, angle in result.placements.values():
            assert isinstance(angle, float)

    def test_allow_rotation_false(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=4,
                                   allow_rotation=False)
        assert isinstance(result, Assembly)

    def test_n_generations_0(self):
        frags, entries = _build_chain(3)
        result = genetic_assembly(frags, entries, n_generations=0, population_size=4)
        assert isinstance(result, Assembly)

    def test_placements_count_equals_frags(self):
        n = 4
        frags, entries = _build_chain(n)
        result = genetic_assembly(frags, entries, n_generations=2, population_size=5)
        assert len(result.placements) == n


# ─── TestMakeIndividual ───────────────────────────────────────────────────────

class TestMakeIndividual:
    def test_returns_tuple_of_2(self):
        fids = np.array([0, 1, 2])
        rots = np.array([0.0, np.pi / 2])
        rng = np.random.RandomState(0)
        ind = _make_individual(fids, rots, rng)
        assert isinstance(ind, tuple) and len(ind) == 2

    def test_order_is_permutation(self):
        fids = np.array([0, 1, 2, 3])
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        rng = np.random.RandomState(0)
        order, _ = _make_individual(fids, rots, rng)
        assert sorted(order.tolist()) == sorted(fids.tolist())

    def test_angles_from_rotations(self):
        fids = np.array([0, 1, 2])
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        rng = np.random.RandomState(1)
        _, angles = _make_individual(fids, rots, rng)
        for a in angles:
            assert any(abs(a - r) < 1e-9 for r in rots)


# ─── TestFitness ──────────────────────────────────────────────────────────────

class TestFitness:
    def test_returns_float(self):
        fids = np.array([0, 1, 2])
        ind = (fids.copy(), np.zeros(3))
        score_map = {(0, 1): 0.5, (1, 2): 0.7}
        assert isinstance(_fitness(ind, score_map, fids), float)

    def test_nonneg(self):
        fids = np.array([0, 1, 2])
        ind = (fids.copy(), np.zeros(3))
        score_map = {(0, 1): 0.5, (1, 2): 0.7}
        assert _fitness(ind, score_map, fids) >= 0.0

    def test_empty_score_map_zero(self):
        fids = np.array([0, 1, 2])
        ind = (fids.copy(), np.zeros(3))
        assert _fitness(ind, {}, fids) == pytest.approx(0.0)

    def test_correct_sum(self):
        fids = np.array([0, 1, 2])
        ind = (np.array([0, 1, 2]), np.zeros(3))
        score_map = {(0, 1): 0.5, (1, 2): 0.7}
        assert _fitness(ind, score_map, fids) == pytest.approx(1.2)

    def test_missing_pair_zero(self):
        fids = np.array([0, 1])
        ind = (np.array([0, 1]), np.zeros(2))
        assert _fitness(ind, {}, fids) == pytest.approx(0.0)


# ─── TestOrderCrossover ───────────────────────────────────────────────────────

class TestOrderCrossover:
    def _parent(self, n, seed):
        rng = np.random.RandomState(seed)
        return (rng.permutation(np.arange(n)), np.zeros(n))

    def test_returns_tuple(self):
        child = _order_crossover(self._parent(4, 0), self._parent(4, 1),
                                  np.random.RandomState(2))
        assert isinstance(child, tuple) and len(child) == 2

    def test_child_is_valid_permutation(self):
        n = 5
        child_order, _ = _order_crossover(self._parent(n, 0), self._parent(n, 1),
                                           np.random.RandomState(3))
        assert sorted(child_order.tolist()) == list(range(n))

    def test_single_element(self):
        p = (np.array([0]), np.array([0.0]))
        child_order, _ = _order_crossover(p, p, np.random.RandomState(0))
        assert child_order[0] == 0

    def test_length_preserved(self):
        n = 6
        child_order, _ = _order_crossover(self._parent(n, 4), self._parent(n, 5),
                                           np.random.RandomState(6))
        assert len(child_order) == n


# ─── TestTournamentSelect ─────────────────────────────────────────────────────

class TestTournamentSelect:
    def test_returns_individual(self):
        pop = [(np.array([0, 1]), np.zeros(2)),
               (np.array([1, 0]), np.zeros(2))]
        scores = np.array([0.3, 0.8])
        result = _tournament_select(pop, scores, k=2, rng=np.random.RandomState(0))
        assert isinstance(result, tuple)

    def test_result_from_population(self):
        pop = [(np.array([i]), np.zeros(1)) for i in range(5)]
        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        result = _tournament_select(pop, scores, k=3, rng=np.random.RandomState(0))
        assert any(np.array_equal(result[0], p[0]) for p in pop)

    def test_k_larger_than_population(self):
        pop = [(np.array([0, 1]), np.zeros(2))]
        scores = np.array([0.5])
        result = _tournament_select(pop, scores, k=100, rng=np.random.RandomState(0))
        assert isinstance(result, tuple)


# ─── TestMutate ───────────────────────────────────────────────────────────────

class TestMutate:
    def test_returns_tuple(self):
        order = np.array([0, 1, 2, 3])
        angles = np.zeros(4)
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        result = _mutate((order, angles), rots, np.random.RandomState(0))
        assert isinstance(result, tuple) and len(result) == 2

    def test_length_preserved(self):
        n = 5
        order = np.arange(n)
        angles = np.zeros(n)
        rots = np.array([0.0, np.pi / 2])
        result_order, result_angles = _mutate((order, angles), rots,
                                              np.random.RandomState(1))
        assert len(result_order) == n and len(result_angles) == n

    def test_permutation_preserved(self):
        n = 5
        order = np.arange(n)
        rots = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        result_order, _ = _mutate((order.copy(), np.zeros(n)), rots,
                                   np.random.RandomState(2))
        assert sorted(result_order.tolist()) == list(range(n))

    def test_single_element_unchanged(self):
        order = np.array([42])
        result_order, _ = _mutate((order, np.zeros(1)),
                                   np.array([0.0]), np.random.RandomState(0))
        assert result_order[0] == 42


# ─── TestBuildScoreMap ────────────────────────────────────────────────────────

class TestBuildScoreMap:
    def test_returns_dict(self):
        frags, entries = _build_chain(3)
        etf = _make_edge_to_frag(frags)
        assert isinstance(_build_score_map(entries, etf), dict)

    def test_keys_are_sorted_pairs(self):
        frags, entries = _build_chain(3)
        etf = _make_edge_to_frag(frags)
        for a, b in _build_score_map(entries, etf).keys():
            assert a <= b

    def test_values_are_floats(self):
        frags, entries = _build_chain(3)
        etf = _make_edge_to_frag(frags)
        for v in _build_score_map(entries, etf).values():
            assert isinstance(v, float)

    def test_empty_entries_empty_map(self):
        assert _build_score_map([], {}) == {}

    def test_same_fragment_skipped(self):
        f0 = _frag(0)
        entry = _entry(f0.edges[0], f0.edges[1], score=0.9)
        etf = {e.edge_id: f0.fragment_id for e in f0.edges}
        assert _build_score_map([entry], etf) == {}

    def test_takes_max_score(self):
        f0 = _frag(0)
        f1 = _frag(1)
        etf = _make_edge_to_frag([f0, f1])
        e_low = _entry(f0.edges[0], f1.edges[0], score=0.3)
        e_high = _entry(f0.edges[0], f1.edges[0], score=0.8)
        result = _build_score_map([e_low, e_high], etf)
        assert list(result.values())[0] == pytest.approx(0.8)
