"""
Тесты для puzzle_reconstruction/assembly/genetic.py

Покрытие:
    genetic_assembly       — пустой вход, 1 фрагмент, базовый N=4, воспроизводимость
    _order_crossover       — сохранение всех id, отсутствие дубликатов, n=1
    _mutate                — корректность перестановки после мутации
    _fitness               — возвращает float ≥ 0
    _build_score_map       — пустые entries, дубликаты, edge-to-frag пропуски
    _tournament_select     — возвращает особь из популяции
    Individual population  — воспроизводимость при одинаковом seed
"""
import math
import numpy as np
import pytest

from puzzle_reconstruction.models import (
    Fragment, Assembly, CompatEntry, EdgeSignature, EdgeSide,
)
from puzzle_reconstruction.assembly.genetic import (
    genetic_assembly,
    _order_crossover,
    _mutate,
    _fitness,
    _build_score_map,
    _tournament_select,
)


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

def _make_edge(edge_id: int, length: float = 100.0) -> EdgeSignature:
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
        contour=np.zeros((10, 2)),
    )
    frag.edges = [_make_edge(fid * 10 + i) for i in range(n_edges)]
    return frag


def _make_entry(ei: EdgeSignature, ej: EdgeSignature,
                score: float) -> CompatEntry:
    return CompatEntry(
        edge_i=ei, edge_j=ej, score=score,
        dtw_dist=0.0, css_sim=score, fd_diff=0.0, text_score=0.0,
    )


@pytest.fixture
def four_fragments():
    return [_make_fragment(i) for i in range(4)]


@pytest.fixture
def entries_4(four_fragments):
    frags = four_fragments
    entries = []
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            ei = frags[i].edges[0]
            ej = frags[j].edges[0]
            score = 0.1 * (i + j + 1)
            entries.append(_make_entry(ei, ej, min(score, 1.0)))
    return entries


# ─── genetic_assembly: граничные случаи ───────────────────────────────────────

class TestGeneticEmpty:
    def test_no_fragments(self):
        result = genetic_assembly([], [], population_size=5, n_generations=2)
        assert isinstance(result, Assembly)
        assert result.fragments == []
        assert result.placements == {}

    def test_single_fragment(self):
        frag = _make_fragment(0)
        result = genetic_assembly([frag], [], population_size=5, n_generations=2)
        assert isinstance(result, Assembly)
        assert len(result.fragments) == 1
        assert 0 in result.placements


# ─── genetic_assembly: базовые свойства ───────────────────────────────────────

class TestGeneticBasic:
    def test_returns_assembly(self, four_fragments, entries_4):
        result = genetic_assembly(
            four_fragments, entries_4,
            population_size=10, n_generations=5, seed=42,
        )
        assert isinstance(result, Assembly)

    def test_all_fragments_placed(self, four_fragments, entries_4):
        result = genetic_assembly(
            four_fragments, entries_4,
            population_size=10, n_generations=5, seed=42,
        )
        for frag in four_fragments:
            assert frag.fragment_id in result.placements

    def test_placements_structure(self, four_fragments, entries_4):
        result = genetic_assembly(
            four_fragments, entries_4,
            population_size=10, n_generations=5, seed=42,
        )
        for fid, (pos, angle) in result.placements.items():
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (2,)
            assert isinstance(angle, float)

    def test_total_score_nonneg(self, four_fragments, entries_4):
        result = genetic_assembly(
            four_fragments, entries_4,
            population_size=10, n_generations=5, seed=42,
        )
        assert result.total_score >= 0.0

    def test_grid_layout(self, four_fragments, entries_4):
        """Фрагменты раскладываются в сетку — позиции различны."""
        result = genetic_assembly(
            four_fragments, entries_4,
            population_size=10, n_generations=5, seed=42,
        )
        positions = [result.placements[f.fragment_id][0]
                     for f in four_fragments]
        # Все позиции уникальны (разные ячейки сетки)
        pos_tuples = [tuple(p) for p in positions]
        assert len(set(pos_tuples)) == len(pos_tuples)


class TestGeneticReproducibility:
    def test_same_seed_same_result(self, four_fragments, entries_4):
        r1 = genetic_assembly(
            four_fragments, entries_4,
            population_size=10, n_generations=5, seed=7,
        )
        r2 = genetic_assembly(
            four_fragments, entries_4,
            population_size=10, n_generations=5, seed=7,
        )
        assert r1.total_score == r2.total_score
        for fid in r1.placements:
            p1, a1 = r1.placements[fid]
            p2, a2 = r2.placements[fid]
            assert np.allclose(p1, p2)
            assert math.isclose(a1, a2)

    def test_different_seeds_may_differ(self, four_fragments, entries_4):
        r1 = genetic_assembly(
            four_fragments, entries_4,
            population_size=20, n_generations=10, seed=1,
        )
        r2 = genetic_assembly(
            four_fragments, entries_4,
            population_size=20, n_generations=10, seed=99,
        )
        # Хотя бы один результат должен быть допустимым
        assert isinstance(r1, Assembly)
        assert isinstance(r2, Assembly)


# ─── _order_crossover ─────────────────────────────────────────────────────────

class TestOrderCrossover:
    def _make_ind(self, order, angles=None):
        n = len(order)
        a = angles if angles is not None else np.zeros(n)
        return (np.array(order, dtype=int), np.array(a, dtype=float))

    def test_preserves_all_ids(self):
        rng = np.random.RandomState(0)
        p1 = self._make_ind([0, 1, 2, 3, 4])
        p2 = self._make_ind([4, 3, 2, 1, 0])
        child_order, _ = _order_crossover(p1, p2, rng)
        assert sorted(child_order.tolist()) == [0, 1, 2, 3, 4]

    def test_no_duplicates(self):
        rng = np.random.RandomState(42)
        ids = list(range(8))
        p1 = self._make_ind(ids)
        p2 = self._make_ind(list(reversed(ids)))
        for _ in range(20):
            child_order, _ = _order_crossover(p1, p2, rng)
            assert len(set(child_order.tolist())) == 8

    def test_single_element(self):
        rng = np.random.RandomState(0)
        p1 = self._make_ind([7])
        p2 = self._make_ind([7])
        child_order, _ = _order_crossover(p1, p2, rng)
        assert child_order[0] == 7

    def test_child_length(self):
        rng = np.random.RandomState(5)
        n = 6
        p1 = self._make_ind(list(range(n)))
        p2 = self._make_ind(list(reversed(range(n))))
        child_order, child_angles = _order_crossover(p1, p2, rng)
        assert len(child_order) == n
        assert len(child_angles) == n


# ─── _mutate ──────────────────────────────────────────────────────────────────

class TestMutate:
    def _make_ind(self, n):
        order = np.arange(n, dtype=int)
        angles = np.zeros(n, dtype=float)
        return (order, angles)

    def test_preserves_fragment_ids(self):
        rng = np.random.RandomState(0)
        ind = self._make_ind(6)
        rotations = np.array([0.0, 1.5707963, 3.1415926, 4.7123889])
        for _ in range(30):
            out = _mutate(ind, rotations, rng)
            assert sorted(out[0].tolist()) == list(range(6))

    def test_angles_from_rotations(self):
        """Углы после мутации должны входить в допустимый набор."""
        rng = np.random.RandomState(1)
        ind = self._make_ind(5)
        rotations = np.array([0.0, 1.5707963, 3.1415926, 4.7123889])
        for _ in range(30):
            out = _mutate(ind, rotations, rng, allow_rotation=True)
            for a in out[1]:
                assert any(math.isclose(a, r, abs_tol=1e-5) for r in rotations)

    def test_single_fragment_unchanged(self):
        rng = np.random.RandomState(0)
        ind = self._make_ind(1)
        rotations = np.array([0.0, 1.5707963])
        out = _mutate(ind, rotations, rng)
        assert out[0][0] == 0


# ─── _fitness ─────────────────────────────────────────────────────────────────

class TestFitness:
    def test_empty_score_map(self):
        ind = (np.array([0, 1, 2], dtype=int), np.zeros(3))
        score = _fitness(ind, {}, np.array([0, 1, 2]))
        assert score == 0.0

    def test_returns_float(self):
        ind = (np.array([0, 1], dtype=int), np.zeros(2))
        score_map = {(0, 1): 0.8}
        score = _fitness(ind, score_map, np.array([0, 1]))
        assert isinstance(score, float)

    def test_sums_adjacent_scores(self):
        ind = (np.array([0, 1, 2], dtype=int), np.zeros(3))
        score_map = {(0, 1): 0.5, (1, 2): 0.3}
        score = _fitness(ind, score_map, np.array([0, 1, 2]))
        assert math.isclose(score, 0.8, abs_tol=1e-6)

    def test_nonnegativity(self, four_fragments, entries_4):
        """Fitness не может быть отрицательным при score ≥ 0."""
        frag_ids = np.array([f.fragment_id for f in four_fragments])
        edge_to_frag = {e.edge_id: f.fragment_id
                        for f in four_fragments for e in f.edges}
        score_map = _build_score_map(entries_4, edge_to_frag)
        ind = (frag_ids.copy(), np.zeros(len(frag_ids)))
        assert _fitness(ind, score_map, frag_ids) >= 0.0


# ─── _build_score_map ─────────────────────────────────────────────────────────

class TestBuildScoreMap:
    def test_empty_entries(self):
        result = _build_score_map([], {})
        assert result == {}

    def test_basic_entry(self):
        f0 = _make_fragment(0)
        f1 = _make_fragment(1)
        ei = f0.edges[0]
        ej = f1.edges[0]
        edge_to_frag = {ei.edge_id: 0, ej.edge_id: 1}
        entry = _make_entry(ei, ej, 0.7)
        result = _build_score_map([entry], edge_to_frag)
        assert (0, 1) in result
        assert math.isclose(result[(0, 1)], 0.7, abs_tol=1e-6)

    def test_takes_max_score(self):
        """Для одной пары фрагментов записывается максимальный score."""
        f0, f1 = _make_fragment(0), _make_fragment(1)
        ei, ej = f0.edges[0], f1.edges[0]
        edge_to_frag = {ei.edge_id: 0, ej.edge_id: 1}
        entries = [
            _make_entry(ei, ej, 0.4),
            _make_entry(ei, ej, 0.9),
            _make_entry(ei, ej, 0.6),
        ]
        result = _build_score_map(entries, edge_to_frag)
        assert math.isclose(result[(0, 1)], 0.9, abs_tol=1e-6)

    def test_missing_edge_skipped(self):
        """Edges без записи в edge_to_frag должны быть пропущены."""
        f0 = _make_fragment(0)
        ei = f0.edges[0]
        orphan = _make_edge(999)  # Нет в edge_to_frag
        entry = _make_entry(ei, orphan, 0.5)
        edge_to_frag = {ei.edge_id: 0}
        result = _build_score_map([entry], edge_to_frag)
        assert result == {}

    def test_same_fragment_skipped(self):
        """Пара ребёр одного фрагмента не добавляется в карту."""
        f0 = _make_fragment(0)
        ei, ej = f0.edges[0], f0.edges[1]
        edge_to_frag = {ei.edge_id: 0, ej.edge_id: 0}
        entry = _make_entry(ei, ej, 0.99)
        result = _build_score_map([entry], edge_to_frag)
        assert result == {}


# ─── _tournament_select ───────────────────────────────────────────────────────

class TestTournamentSelect:
    def _make_population(self, size, n=4):
        rng = np.random.RandomState(0)
        return [(rng.permutation(n), np.zeros(n)) for _ in range(size)]

    def test_returns_member(self):
        pop = self._make_population(10)
        scores = np.arange(10, dtype=float)
        rng = np.random.RandomState(0)
        chosen = _tournament_select(pop, scores, k=3, rng=rng)
        assert any(np.array_equal(chosen[0], p[0]) for p in pop)

    def test_k_larger_than_population(self):
        """k > len(population) не должен вызывать ошибку."""
        pop = self._make_population(3)
        scores = np.array([0.1, 0.5, 0.9])
        rng = np.random.RandomState(0)
        chosen = _tournament_select(pop, scores, k=100, rng=rng)
        assert any(np.array_equal(chosen[0], p[0]) for p in pop)

    def test_prefers_higher_score(self):
        """При большом числе попыток чаще выбирается особь с высоким score."""
        pop = self._make_population(5)
        scores = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        rng = np.random.RandomState(42)
        chosen_indices = []
        for _ in range(200):
            c = _tournament_select(pop, scores, k=3, rng=rng)
            for i, p in enumerate(pop):
                if np.array_equal(c[0], p[0]):
                    chosen_indices.append(i)
                    break
        # Лучшая особь (индекс 4) должна встречаться чаще всего
        from collections import Counter
        cnt = Counter(chosen_indices)
        assert cnt[4] > 50, f"Лучшая особь выбирается редко: {cnt}"
