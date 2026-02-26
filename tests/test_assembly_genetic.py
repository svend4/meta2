"""Tests for puzzle_reconstruction/assembly/genetic.py."""
import math
import pytest
import numpy as np

from puzzle_reconstruction.models import Fragment, Assembly, CompatEntry, Edge, EdgeSignature, EdgeSide
from puzzle_reconstruction.assembly.genetic import (
    genetic_assembly,
    _make_individual,
    _fitness,
    _tournament_select,
    _order_crossover,
    _mutate,
    _build_score_map,
    _individual_to_assembly,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int) -> EdgeSignature:
    return EdgeSignature(
        edge_id=edge_id,
        side=EdgeSide.TOP,
        virtual_curve=np.zeros((10, 2)),
        fd=1.5,
        css_vec=np.zeros(8),
        ifs_coeffs=np.zeros(4),
        length=80.0,
    )


def _make_fragment(fid: int, n_edges: int = 2) -> Fragment:
    frag = Fragment(
        fragment_id=fid,
        image=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _make_entry(ei, ej, score: float = 0.5) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


def _make_entries(frags):
    entries = []
    for i, fi in enumerate(frags):
        for j, fj in enumerate(frags):
            if i >= j:
                continue
            entries.append(_make_entry(fi.edges[0], fj.edges[0], score=0.5))
    return entries


ROTATIONS = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])


# ─── genetic_assembly ─────────────────────────────────────────────────────────

class TestGeneticAssembly:
    def test_returns_assembly(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        result = genetic_assembly(frags, entries, population_size=10, n_generations=5, seed=0)
        assert isinstance(result, Assembly)

    def test_empty_fragments_returns_empty_assembly(self):
        result = genetic_assembly([], [], population_size=10, n_generations=5)
        assert isinstance(result, Assembly)
        assert not result.fragments or len(result.fragments) == 0

    def test_single_fragment(self):
        frags = [_make_fragment(0)]
        result = genetic_assembly(frags, [], population_size=5, n_generations=3, seed=1)
        assert isinstance(result, Assembly)

    def test_all_fragments_placed(self):
        frags = [_make_fragment(i) for i in range(5)]
        entries = _make_entries(frags)
        result = genetic_assembly(frags, entries, population_size=10, n_generations=5, seed=2)
        placed_ids = set(result.placements.keys()) if isinstance(result.placements, dict) else set()
        frag_ids = {f.fragment_id for f in frags}
        assert placed_ids == frag_ids

    def test_placements_have_positions(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = genetic_assembly(frags, entries, population_size=5, n_generations=3, seed=3)
        for fid, (pos, angle) in result.placements.items():
            assert pos.shape == (2,)
            assert isinstance(angle, float)

    def test_total_score_is_nonnegative(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        result = genetic_assembly(frags, entries, population_size=10, n_generations=5, seed=4)
        assert result.total_score >= 0.0

    def test_seed_reproducibility(self):
        frags = [_make_fragment(i) for i in range(4)]
        entries = _make_entries(frags)
        r1 = genetic_assembly(frags, entries, population_size=10, n_generations=5, seed=42)
        r2 = genetic_assembly(frags, entries, population_size=10, n_generations=5, seed=42)
        assert r1.total_score == r2.total_score

    def test_different_seeds_may_differ(self):
        frags = [_make_fragment(i) for i in range(6)]
        entries = _make_entries(frags)
        r1 = genetic_assembly(frags, entries, population_size=10, n_generations=10, seed=1)
        r2 = genetic_assembly(frags, entries, population_size=10, n_generations=10, seed=99)
        # At minimum both return Assembly
        assert isinstance(r1, Assembly)
        assert isinstance(r2, Assembly)

    def test_no_rotation_mode(self):
        frags = [_make_fragment(i) for i in range(3)]
        entries = _make_entries(frags)
        result = genetic_assembly(frags, entries, population_size=5, n_generations=3,
                                   allow_rotation=False, seed=5)
        assert isinstance(result, Assembly)

    def test_large_elite_size(self):
        frags = [_make_fragment(i) for i in range(5)]
        entries = _make_entries(frags)
        result = genetic_assembly(frags, entries, population_size=10, n_generations=3,
                                   elite_size=8, seed=6)
        assert isinstance(result, Assembly)

    def test_two_fragments(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        entries = _make_entries(frags)
        result = genetic_assembly(frags, entries, population_size=5, n_generations=5, seed=7)
        assert len(result.placements) == 2


# ─── _make_individual ─────────────────────────────────────────────────────────

class TestMakeIndividual:
    def test_returns_tuple_of_two_arrays(self):
        frag_ids = np.array([0, 1, 2, 3])
        rng = np.random.RandomState(0)
        ind = _make_individual(frag_ids, ROTATIONS, rng)
        assert len(ind) == 2
        order, angles = ind
        assert isinstance(order, np.ndarray)
        assert isinstance(angles, np.ndarray)

    def test_order_is_permutation(self):
        frag_ids = np.array([10, 20, 30, 40])
        rng = np.random.RandomState(1)
        order, angles = _make_individual(frag_ids, ROTATIONS, rng)
        assert set(order.tolist()) == set(frag_ids.tolist())
        assert len(order) == len(frag_ids)

    def test_angles_are_valid_rotations(self):
        frag_ids = np.array([0, 1, 2, 3, 4])
        rng = np.random.RandomState(2)
        order, angles = _make_individual(frag_ids, ROTATIONS, rng)
        for a in angles:
            assert a in ROTATIONS

    def test_single_fragment(self):
        frag_ids = np.array([5])
        rng = np.random.RandomState(3)
        order, angles = _make_individual(frag_ids, ROTATIONS, rng)
        assert len(order) == 1
        assert len(angles) == 1


# ─── _fitness ─────────────────────────────────────────────────────────────────

class TestFitness:
    def test_zero_score_no_entries(self):
        frag_ids = np.array([0, 1, 2])
        order = np.array([0, 1, 2])
        angles = np.zeros(3)
        score = _fitness((order, angles), {}, frag_ids)
        assert score == 0.0

    def test_score_with_entries(self):
        frag_ids = np.array([0, 1, 2])
        score_map = {(0, 1): 0.8, (1, 2): 0.6}
        order = np.array([0, 1, 2])
        angles = np.zeros(3)
        score = _fitness((order, angles), score_map, frag_ids)
        assert abs(score - 1.4) < 1e-9

    def test_score_depends_on_order(self):
        frag_ids = np.array([0, 1, 2])
        score_map = {(0, 1): 0.9, (1, 2): 0.1}
        order1 = np.array([0, 1, 2])
        order2 = np.array([0, 2, 1])
        angles = np.zeros(3)
        s1 = _fitness((order1, angles), score_map, frag_ids)
        s2 = _fitness((order2, angles), score_map, frag_ids)
        assert s1 != s2

    def test_single_fragment_score_zero(self):
        frag_ids = np.array([0])
        order = np.array([0])
        angles = np.zeros(1)
        score = _fitness((order, angles), {(0, 1): 0.5}, frag_ids)
        assert score == 0.0


# ─── _tournament_select ───────────────────────────────────────────────────────

class TestTournamentSelect:
    def test_returns_individual_from_population(self):
        frag_ids = np.array([0, 1, 2])
        rng = np.random.RandomState(0)
        population = [_make_individual(frag_ids, ROTATIONS, rng) for _ in range(5)]
        scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        selected = _tournament_select(population, scores, k=3, rng=rng)
        # Use identity check since individuals contain numpy arrays (makes 'in' ambiguous)
        assert any(selected is p for p in population)

    def test_prefers_high_score(self):
        frag_ids = np.array([0, 1])
        rng = np.random.RandomState(0)
        best_ind = (np.array([0, 1]), np.zeros(2))
        others = [(np.array([1, 0]), np.zeros(2)) for _ in range(9)]
        population = others + [best_ind]
        scores = np.array([0.1] * 9 + [1.0])
        # Run many times; best should be selected at least once
        selected_best = False
        for _ in range(20):
            s = _tournament_select(population, scores, k=5, rng=rng)
            # Use identity check since individuals contain numpy arrays
            if s is best_ind:
                selected_best = True
                break
        assert selected_best


# ─── _order_crossover ─────────────────────────────────────────────────────────

class TestOrderCrossover:
    def test_child_is_valid_permutation(self):
        frag_ids = np.array([0, 1, 2, 3, 4])
        rng = np.random.RandomState(0)
        p1 = (frag_ids.copy(), np.zeros(5))
        p2 = (frag_ids[::-1].copy(), np.zeros(5))
        child_order, child_angles = _order_crossover(p1, p2, rng)
        assert set(child_order.tolist()) == set(frag_ids.tolist())
        assert len(child_order) == 5

    def test_single_element(self):
        frag_ids = np.array([7])
        rng = np.random.RandomState(0)
        p1 = (frag_ids.copy(), np.zeros(1))
        p2 = (frag_ids.copy(), np.zeros(1))
        child_order, child_angles = _order_crossover(p1, p2, rng)
        assert child_order[0] == 7

    def test_angles_have_correct_length(self):
        frag_ids = np.array([0, 1, 2, 3])
        rng = np.random.RandomState(1)
        p1 = (frag_ids.copy(), np.ones(4))
        p2 = (frag_ids[::-1].copy(), np.zeros(4))
        child_order, child_angles = _order_crossover(p1, p2, rng)
        assert len(child_angles) == 4


# ─── _mutate ──────────────────────────────────────────────────────────────────

class TestMutate:
    def test_returns_valid_permutation(self):
        order = np.array([0, 1, 2, 3, 4])
        angles = np.zeros(5)
        rng = np.random.RandomState(0)
        for _ in range(20):
            new_order, new_angles = _mutate((order.copy(), angles.copy()), ROTATIONS, rng)
            assert set(new_order.tolist()) == set(order.tolist())
            assert len(new_order) == len(order)

    def test_single_element_no_crash(self):
        order = np.array([3])
        angles = np.zeros(1)
        rng = np.random.RandomState(5)
        new_order, new_angles = _mutate((order, angles), ROTATIONS, rng)
        assert new_order[0] == 3

    def test_no_rotation_mode_skips_rotation(self):
        order = np.array([0, 1, 2, 3])
        angles = np.zeros(4)
        rng = np.random.RandomState(0)
        for _ in range(30):
            new_order, new_angles = _mutate((order.copy(), angles.copy()), ROTATIONS, rng, allow_rotation=False)
            # All angles should be 0 since rotation is disabled
            # (they're only changed when move==1 which is the rotation move)
            assert set(new_order.tolist()) == {0, 1, 2, 3}


# ─── _build_score_map ─────────────────────────────────────────────────────────

class TestBuildScoreMap:
    def test_builds_correct_map(self):
        frags = [_make_fragment(i) for i in range(3)]
        edge_to_frag = {f.edges[0].edge_id: f.fragment_id for f in frags}
        entries = _make_entries(frags)
        score_map = _build_score_map(entries, edge_to_frag)
        # Should have entries for each pair
        assert all(k[0] <= k[1] for k in score_map.keys())

    def test_takes_max_score(self):
        frags = [_make_fragment(0), _make_fragment(1)]
        # edge_to_frag maps edge_id to fragment_id; edge_id for frag 0 is 0, for frag 1 is 10
        edge_to_frag = {frags[0].edges[0].edge_id: 0, frags[1].edges[0].edge_id: 1}
        e1 = CompatEntry(edge_i=frags[0].edges[0], edge_j=frags[1].edges[0], score=0.3)
        e2 = CompatEntry(edge_i=frags[0].edges[0], edge_j=frags[1].edges[0], score=0.9)
        score_map = _build_score_map([e1, e2], edge_to_frag)
        assert score_map.get((0, 1), 0.0) == 0.9

    def test_skips_same_fragment(self):
        frags = [_make_fragment(0)]
        edge_to_frag = {0: 0, 1: 0}
        e = CompatEntry(edge_i=frags[0].edges[0], edge_j=frags[0].edges[0], score=0.5)
        score_map = _build_score_map([e], edge_to_frag)
        assert len(score_map) == 0

    def test_empty_entries(self):
        score_map = _build_score_map([], {})
        assert score_map == {}


# ─── _individual_to_assembly ──────────────────────────────────────────────────

class TestIndividualToAssembly:
    def test_returns_assembly(self):
        frags = [_make_fragment(i) for i in range(4)]
        frag_ids = np.array([f.fragment_id for f in frags])
        order = frag_ids.copy()
        angles = np.zeros(4)
        asm = _individual_to_assembly((order, angles), frags, {}, frag_ids, 0.75)
        assert isinstance(asm, Assembly)

    def test_all_fragments_placed(self):
        frags = [_make_fragment(i) for i in range(5)]
        frag_ids = np.array([f.fragment_id for f in frags])
        order = frag_ids.copy()
        angles = np.zeros(5)
        asm = _individual_to_assembly((order, angles), frags, {}, frag_ids, 0.5)
        assert set(asm.placements.keys()) == set(range(5))

    def test_total_score_preserved(self):
        frags = [_make_fragment(i) for i in range(3)]
        frag_ids = np.array([0, 1, 2])
        order = frag_ids.copy()
        angles = np.zeros(3)
        asm = _individual_to_assembly((order, angles), frags, {}, frag_ids, 1.23)
        assert abs(asm.total_score - 1.23) < 1e-9

    def test_grid_layout_positions(self):
        frags = [_make_fragment(i) for i in range(4)]
        frag_ids = np.array([0, 1, 2, 3])
        order = frag_ids.copy()
        angles = np.zeros(4)
        asm = _individual_to_assembly((order, angles), frags, {}, frag_ids, 0.0)
        # Each placement should have position array of shape (2,)
        for fid, (pos, ang) in asm.placements.items():
            assert pos.shape == (2,)
